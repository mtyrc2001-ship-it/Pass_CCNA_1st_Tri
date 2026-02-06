# ui/flashcards.py
from __future__ import annotations

import csv
import io
import json
import random
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from config import CCNA_BLUEPRINT, GEMINI_API_KEY
from db.database import (
    add_flashcard,
    bulk_add_flashcards,
    delete_flashcard,
    list_books,
    list_flashcards,
    update_flashcard_progress,
    get_questions,
)

BOOKS_DIR = Path("data/books")
DOMAINS = list(CCNA_BLUEPRINT.keys())


# ----------------------------
# Helpers: PDF + chapters
# ----------------------------
def _require_pymupdf():
    try:
        import fitz  # PyMuPDF
        return fitz
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Missing dependency: PyMuPDF. Install with: pip install pymupdf") from e


def pdf_text_from_path(path: Path) -> str:
    fitz = _require_pymupdf()
    doc = fitz.open(str(path))
    try:
        return "".join(page.get_text() for page in doc)
    finally:
        doc.close()


def split_into_chapters(text: str) -> List[Tuple[str, str]]:
    pattern = re.compile(r"(?im)^\s*(chapter|unit)\s+\d+.*$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        return [("Full Document", text)]

    chunks: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = m.group(0).strip()
        chunks.append((title, text[start:end]))
    return chunks


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ----------------------------
# Helpers: weak domains
# ----------------------------
def compute_domain_accuracy() -> Dict[str, Dict[str, float]]:
    qs = get_questions() or []
    stats: Dict[str, Dict[str, float]] = {d: {"seen": 0, "correct": 0, "accuracy": 0.0} for d in DOMAINS}

    for q in qs:
        d = q.get("domain")
        if d not in stats:
            continue
        seen = int(q.get("seen", 0) or 0)
        correct = int(q.get("correct", 0) or 0)
        stats[d]["seen"] += seen
        stats[d]["correct"] += correct

    for d in DOMAINS:
        seen = stats[d]["seen"]
        correct = stats[d]["correct"]
        stats[d]["accuracy"] = (correct / seen) if seen > 0 else 0.0

    return stats


def weak_domains(threshold: float = 0.75) -> List[str]:
    stats = compute_domain_accuracy()
    weak = []
    for d, v in stats.items():
        if int(v["seen"]) > 0 and float(v["accuracy"]) < threshold:
            weak.append(d)
    return weak


# ----------------------------
# Helpers: AI generation (Gemini)
# ----------------------------
def _clean_json(text: str) -> str:
    t = text.strip()
    t = t.replace("```json", "").replace("```", "").strip()
    return t


def _try_parse_json_list(raw: str) -> List[Dict[str, Any]]:
    cleaned = _clean_json(raw)
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    except Exception:
        pass

    m = re.search(r"(\[\s*\{.*\}\s*\])", cleaned, flags=re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            if isinstance(data, list):
                return [x for x in data if isinstance(x, dict)]
        except Exception:
            return []

    return []


def generate_flashcards_with_gemini(
    chapter_text: str,
    n: int,
    default_domain: str,
    chapter_id: Optional[str] = None,
    source_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        raise RuntimeError("Gemini API key not set. Set GEMINI_API_KEY in your environment or config.py.")

    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
Create {n} CCNA flashcards from the text below.

Rules:
- Return JSON ONLY (no markdown fences).
- Output must be a JSON array of objects.
- Each object must include: front, back
- Optionally include: domain (one of: {DOMAINS})
- Keep fronts short (1 line). Keep backs concise (1-3 lines).
- Use domain "{default_domain}" when unsure.

TEXT:
{chapter_text[:12000]}
"""

    raw = model.generate_content(prompt).text
    cards = _try_parse_json_list(raw)

    out: List[Dict[str, Any]] = []
    for c in cards:
        front = (c.get("front") or c.get("f") or "").strip()
        back = (c.get("back") or c.get("b") or "").strip()
        dom = (c.get("domain") or default_domain or "Network Fundamentals").strip()
        if dom not in DOMAINS:
            dom = default_domain if default_domain in DOMAINS else "Network Fundamentals"

        if not front or not back:
            continue

        out.append(
            {
                "front": front,
                "back": back,
                "domain": dom,
                "chapter_id": chapter_id,
                "source_name": source_name,
                "created_at": _now_str(),
            }
        )
    return out


# ----------------------------
# Anki import (.apkg) — enhanced
# ----------------------------
def _require_ankipandas():
    try:
        from ankipandas import Collection  # type: ignore
        return Collection
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Missing dependency: ankipandas. Install with: pip install ankipandas") from e


def _anki_html_to_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.IGNORECASE)
    s = re.sub(r"</p\s*>", "\n", s, flags=re.IGNORECASE)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


_CLOZE_RE = re.compile(r"\{\{c(\d+)::(.*?)(::(.*?))?\}\}", flags=re.IGNORECASE | re.DOTALL)


def _extract_cloze_cards(text: str) -> List[Tuple[str, str]]:
    if not text or "{{c" not in text.lower():
        return []

    out: List[Tuple[str, str]] = []
    matches = list(_CLOZE_RE.finditer(text))
    for m in matches:
        answer = (m.group(2) or "").strip()
        hint = (m.group(4) or "").strip()
        if not answer:
            continue

        q = text[: m.start()] + "____" + text[m.end() :]
        q = _CLOZE_RE.sub(lambda mm: mm.group(2) or "", q)
        q = _anki_html_to_text(q)

        a = answer
        if hint:
            a = f"{answer}\n(Hint: {hint})"
        a = _anki_html_to_text(a)

        if q and a:
            out.append((q, a))

    return out


def _tag_to_domain(tag: str) -> Optional[str]:
    t = (tag or "").strip().lower()
    if not t:
        return None

    domain_keywords = {
        "Network Fundamentals": ["fundamentals", "nf", "basics", "osi", "tcpip", "subnet", "ipv4", "ipv6"],
        "Network Access": ["access", "na", "switching", "vlan", "stp", "etherchannel", "wireless"],
        "IP Connectivity": ["connectivity", "ipc", "routing", "ospf", "static", "eigrp", "rip"],
        "IP Services": ["services", "ips", "dhcp", "dns", "nat", "ntp", "snmp", "syslog", "qos"],
        "Security Fundamentals": ["security", "sec", "acl", "aaa", "vpn", "8021x", "wpa"],
        "Automation and Programmability": ["automation", "programmability", "api", "rest", "json", "yaml", "netconf", "restconf"],
        "Wireless": ["wireless", "wlc", "capwap", "ssid", "80211", "rf"],
    }

    for d in DOMAINS:
        dl = d.lower()
        if dl.replace(" ", "") in t.replace("_", "").replace("-", ""):
            return d

    for d, kws in domain_keywords.items():
        for kw in kws:
            if kw in t:
                return d

    return None


def _best_domain_from_tags(tags_str: str, default_domain: str) -> str:
    tags = [x for x in (tags_str or "").strip().split() if x]
    for tg in tags:
        d = _tag_to_domain(tg)
        if d:
            return d
    return default_domain if default_domain in DOMAINS else "Network Fundamentals"


def _try_get_model_field_names(col) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}
    try:
        models = getattr(col, "models", None)
        if models is None:
            return out

        if isinstance(models, dict):
            for mid, m in models.items():
                flds = m.get("flds") if isinstance(m, dict) else None
                if isinstance(flds, list):
                    names = []
                    for f in flds:
                        if isinstance(f, dict) and f.get("name"):
                            names.append(str(f["name"]))
                    if names:
                        out[int(mid)] = names
            return out

    except Exception:
        return {}

    return out


def parse_anki_apkg_enhanced(
    file_bytes: bytes,
    *,
    default_domain: str,
    source_label: str,
    front_field_idx: int,
    back_field_idx: int,
    cloze_mode: str,
    cloze_field_idx: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    Collection = _require_ankipandas()

    with tempfile.TemporaryDirectory() as td:
        apkg_path = Path(td) / "deck.apkg"
        apkg_path.write_bytes(file_bytes)

        col = Collection(str(apkg_path))
        notes = col.notes

        if "flds" not in notes.columns:
            raise ValueError("Could not find 'flds' in Anki notes. This deck format may be unsupported.")

        model_fields = _try_get_model_field_names(col)

        out: List[Dict[str, Any]] = []
        meta = {
            "notes_total": int(len(notes)),
            "cards_built": 0,
            "model_fields_available": bool(model_fields),
        }

        for _, row in notes.iterrows():
            flds = str(row.get("flds") or "")
            parts = flds.split("\x1f")
            if len(parts) < 1:
                continue

            tags_str = str(row.get("tags") or "") if "tags" in notes.columns else ""

            def get_field(i: int) -> str:
                if i < 0 or i >= len(parts):
                    return ""
                return _anki_html_to_text(parts[i])

            dom = _best_domain_from_tags(tags_str, default_domain)

            front_raw = get_field(front_field_idx)
            back_raw = get_field(back_field_idx)
            cloze_text = get_field(cloze_field_idx)

            if cloze_mode == "Always cloze":
                is_cloze = True
            elif cloze_mode == "Never cloze":
                is_cloze = False
            else:
                is_cloze = ("{{c" in (cloze_text or "").lower())

            if is_cloze:
                pairs = _extract_cloze_cards(cloze_text)
                if pairs:
                    for (q_txt, a_txt) in pairs:
                        out.append(
                            {
                                "front": q_txt,
                                "back": a_txt,
                                "domain": dom,
                                "chapter_id": None,
                                "source_name": source_label,
                                "created_at": _now_str(),
                            }
                        )
                    continue

            if front_raw and back_raw:
                out.append(
                    {
                        "front": front_raw,
                        "back": back_raw,
                        "domain": dom,
                        "chapter_id": None,
                        "source_name": source_label,
                        "created_at": _now_str(),
                    }
                )

        meta["cards_built"] = len(out)

        try:
            if model_fields and "mid" in notes.columns:
                mids = notes["mid"].dropna().unique().tolist()
                if mids:
                    mid0 = int(mids[0])
                    meta["example_field_names"] = model_fields.get(mid0, [])
        except Exception:
            pass

        return out, meta


# ----------------------------
# Session deck
# ----------------------------
def _init_state():
    if "deck" not in st.session_state:
        st.session_state.deck = []
    if "deck_i" not in st.session_state:
        st.session_state.deck_i = 0
    if "flip" not in st.session_state:
        st.session_state.flip = False
    if "active_filter" not in st.session_state:
        st.session_state.active_filter = {}


def _load_deck(domain: Optional[str], weak_only: bool, shuffle: bool):
    w = weak_domains() if weak_only else None
    dom_filter = None if (domain in (None, "All Domains")) else domain

    cards = list_flashcards(domain=dom_filter, domains_in=w) or []
    if shuffle:
        random.shuffle(cards)

    st.session_state.deck = cards
    st.session_state.deck_i = 0
    st.session_state.flip = False
    st.session_state.active_filter = {"domain": dom_filter, "weak_only": weak_only, "shuffle": shuffle}


# ----------------------------
# CSV import
# ----------------------------
def _parse_csv(file_bytes: bytes) -> List[Dict[str, Any]]:
    text = file_bytes.decode("utf-8", errors="ignore")
    buf = io.StringIO(text)
    reader = csv.DictReader(buf)

    out: List[Dict[str, Any]] = []
    for row in reader:
        front = (row.get("front") or row.get("f") or "").strip()
        back = (row.get("back") or row.get("b") or "").strip()
        domain = (row.get("domain") or "").strip() or "Network Fundamentals"
        if domain not in DOMAINS:
            domain = "Network Fundamentals"

        if not front or not back:
            continue

        out.append(
            {
                "front": front,
                "back": back,
                "domain": domain,
                "chapter_id": (row.get("chapter_id") or "").strip() or None,
                "source_name": (row.get("source_name") or "").strip() or None,
                "created_at": _now_str(),
            }
        )
    return out


# ----------------------------
# UI
# ----------------------------
def show_flashcards() -> None:
    _init_state()
    st.header("🃏 Flashcards")

    tabs = st.tabs(["Study", "Add / Import", "Generate from Books"])

    # =========================
    # TAB 1: Study
    # =========================
    with tabs[0]:
        stats = compute_domain_accuracy()
        weak = weak_domains()

        colA, colB, colC, colD = st.columns([2, 2, 1, 1])
        with colA:
            domain = st.selectbox("Domain", ["All Domains"] + DOMAINS, index=0)
        with colB:
            weak_only = st.checkbox(
                "Weak topics only",
                value=False,
                help=f"Weak domains detected: {', '.join(weak) if weak else 'None yet'}",
            )
        with colC:
            shuffle = st.checkbox("Shuffle", value=True)
        with colD:
            if st.button("Reload deck"):
                _load_deck(domain=domain, weak_only=weak_only, shuffle=shuffle)
                st.rerun()

        desired = {"domain": None if domain == "All Domains" else domain, "weak_only": weak_only, "shuffle": shuffle}
        if not st.session_state.deck or st.session_state.active_filter != desired:
            _load_deck(domain=domain, weak_only=weak_only, shuffle=shuffle)

        if not st.session_state.deck:
            st.info("No flashcards found for this filter. Add/import/generate some in the other tabs.")
        else:
            i = max(0, min(st.session_state.deck_i, len(st.session_state.deck) - 1))
            st.session_state.deck_i = i

            card = st.session_state.deck[i]
            front = card.get("front", "")
            back = card.get("back", "")
            dom = card.get("domain", "—")
            chapter_id = card.get("chapter_id") or "—"
            source_name = card.get("source_name") or "—"
            seen = int(card.get("seen", 0) or 0)
            correct = int(card.get("correct", 0) or 0)

            st.caption(f"Card {i+1}/{len(st.session_state.deck)} | Domain: {dom} | Chapter: {chapter_id} | Source: {source_name}")
            st.caption(
                f"Performance: seen={seen}, correct={correct} ({(correct/seen*100):.1f}%)" if seen > 0 else "Performance: not studied yet"
            )

            st.markdown("---")
            st.subheader(back if st.session_state.flip else front)

            col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])
            with col1:
                if st.button("Flip"):
                    st.session_state.flip = not st.session_state.flip
                    st.rerun()
            with col2:
                if st.button("◀ Prev", disabled=(i <= 0)):
                    st.session_state.deck_i -= 1
                    st.session_state.flip = False
                    st.rerun()
            with col3:
                if st.button("Next ▶", disabled=(i >= len(st.session_state.deck) - 1)):
                    st.session_state.deck_i += 1
                    st.session_state.flip = False
                    st.rerun()
            with col4:
                if st.button("🔀 Random"):
                    st.session_state.deck_i = random.randint(0, len(st.session_state.deck) - 1)
                    st.session_state.flip = False
                    st.rerun()
            with col5:
                st.write("")

            st.markdown("### Grade this card")
            colK, colM, colDel = st.columns([1, 1, 1])
            with colK:
                if st.button("✅ I knew it"):
                    update_flashcard_progress(int(card["id"]), True)
                    _load_deck(domain=domain, weak_only=weak_only, shuffle=shuffle)
                    st.rerun()
            with colM:
                if st.button("❌ I missed it"):
                    update_flashcard_progress(int(card["id"]), False)
                    _load_deck(domain=domain, weak_only=weak_only, shuffle=shuffle)
                    st.rerun()
            with colDel:
                if st.button("🗑️ Delete this card"):
                    delete_flashcard(int(card["id"]))
                    _load_deck(domain=domain, weak_only=weak_only, shuffle=shuffle)
                    st.rerun()

            st.markdown("---")
            st.caption("Weak-topics mode uses your **question accuracy** per domain (Practice/Exam).")

            with st.expander("📊 Domain accuracy snapshot"):
                df = []
                for d in DOMAINS:
                    seen_d = int(stats[d]["seen"])
                    acc = float(stats[d]["accuracy"])
                    df.append({"Domain": d, "Seen": seen_d, "Accuracy (%)": round(acc * 100, 1)})
                st.dataframe(df, use_container_width=True)

    # =========================
    # TAB 2: Add / Import
    # =========================
    with tabs[1]:
        st.subheader("Add a flashcard (manual)")

        with st.form("add_flashcard_form", clear_on_submit=True):
            front = st.text_input("Front (question / prompt)")
            back = st.text_area("Back (answer / explanation)")
            domain = st.selectbox("Domain", DOMAINS, index=0)
            chapter_id = st.text_input("Chapter ID (optional)", value="")
            source_name = st.text_input("Source name (optional)", value="")
            submitted = st.form_submit_button("Add flashcard")

        if submitted:
            if not front.strip() or not back.strip():
                st.error("Front and Back are required.")
            else:
                add_flashcard(
                    front=front.strip(),
                    back=back.strip(),
                    domain=domain,
                    chapter_id=chapter_id.strip() or None,
                    source_name=source_name.strip() or None,
                    created_at=_now_str(),
                )
                st.success("Flashcard added.")

        st.markdown("---")

        st.subheader("Import flashcards from CSV")
        st.caption("CSV columns: front, back, domain (optional), chapter_id (optional), source_name (optional)")
        csv_file = st.file_uploader("Upload CSV", type=["csv"], key="fc_csv")
        if csv_file:
            try:
                cards = _parse_csv(csv_file.getvalue())
                if not cards:
                    st.warning("No valid rows found. Make sure your CSV has 'front' and 'back' columns.")
                else:
                    st.write(f"Found {len(cards)} cards.")
                    if st.button("Import CSV into database", key="fc_csv_import"):
                        bulk_add_flashcards(cards)
                        st.success(f"Imported {len(cards)} flashcards.")
            except Exception as e:
                st.error(f"Could not parse CSV: {e}")

        st.markdown("---")

        st.subheader("Import Anki deck (.apkg) — Cloze + field mapping + tags→domain")
        st.caption("Requires: pip install ankipandas")

        anki_file = st.file_uploader("Upload .apkg", type=["apkg"], key="fc_anki")
        if anki_file:
            anki_default_domain = st.selectbox(
                "Fallback domain (used when no tag match)",
                DOMAINS,
                index=0,
                key="anki_dom",
            )

            st.markdown("#### Field mapping")
            st.caption("Choose which Anki fields become Front/Back. Indices start at 0. Most basic decks: Front=0, Back=1.")

            front_idx = st.number_input("Front field index", min_value=0, max_value=50, value=0, step=1, key="anki_front_idx")
            back_idx = st.number_input("Back field index", min_value=0, max_value=50, value=1, step=1, key="anki_back_idx")

            st.markdown("#### Cloze handling")
            cloze_mode = st.selectbox(
                "Cloze mode",
                ["Auto (detect cloze)", "Always cloze", "Never cloze"],
                index=0,
                key="anki_cloze_mode",
            )
            cloze_field_idx = st.number_input(
                "Cloze field index (used when cloze is detected/forced)",
                min_value=0,
                max_value=50,
                value=0,
                step=1,
                help="Usually same as Front field. This is the field containing {{c1::...}} markup.",
                key="anki_cloze_idx",
            )

            if st.button("Preview Anki import", key="anki_preview_btn"):
                try:
                    source_label = f"Anki: {anki_file.name}"
                    cards, meta = parse_anki_apkg_enhanced(
                        anki_file.getvalue(),
                        default_domain=anki_default_domain,
                        source_label=source_label,
                        front_field_idx=int(front_idx),
                        back_field_idx=int(back_idx),
                        cloze_mode=str(cloze_mode),
                        cloze_field_idx=int(cloze_field_idx),
                    )

                    st.session_state._anki_preview_cards = cards
                    st.session_state._anki_preview_meta = meta

                    if not cards:
                        st.warning("No usable cards found with this configuration. Try different field indices or cloze settings.")
                    else:
                        st.success(f"Preview built {len(cards)} cards.")
                    st.rerun()

                except Exception as e:
                    st.error(str(e))

            meta = st.session_state.get("_anki_preview_meta") or {}
            preview_cards = st.session_state.get("_anki_preview_cards") or []

            if meta:
                st.caption(
                    f"Notes scanned: {meta.get('notes_total','?')} | Cards built: {meta.get('cards_built','?')} | "
                    f"Field-name metadata available: {meta.get('model_fields_available', False)}"
                )
                if meta.get("example_field_names"):
                    st.caption("Example field names (from first model found): " + ", ".join(meta["example_field_names"]))

            if preview_cards:
                st.markdown("##### Preview (first 25)")
                st.dataframe(
                    [{"domain": c["domain"], "front": c["front"], "back": c["back"]} for c in preview_cards[:25]],
                    use_container_width=True,
                )

                if st.button("Import Anki deck into database", key="anki_import_btn"):
                    bulk_add_flashcards(preview_cards)
                    st.success(f"Imported {len(preview_cards)} Anki cards.")
                    st.session_state._anki_preview_cards = []
                    st.session_state._anki_preview_meta = {}
                    st.rerun()

    # =========================
    # TAB 3: Generate from Books
    # =========================
    with tabs[2]:
        st.subheader("Generate flashcards from a Books Library chapter (AI)")

        books = list_books() or []
        library_filenames = [b.get("filename") for b in books if b.get("filename")]

        if not library_filenames:
            st.info("No books in library yet. Add PDFs in the Books Library tab first.")
        else:
            book = st.selectbox("Pick a book", options=library_filenames, key="gen_book_pick")
            path = BOOKS_DIR / book
            if not path.exists():
                st.error("Book PDF not found in data/books.")
            else:
                chapter_default_domain = st.selectbox("Default domain to tag generated cards", DOMAINS, index=0, key="gen_default_dom")
                n_cards = st.slider("How many flashcards to generate", 5, 50, 20, key="gen_n_cards")
                max_chars = st.slider("Chapter text limit for AI (chars)", 2000, 20000, 12000, step=1000, key="gen_max_chars")

                if st.button("Load chapters from this book", key="gen_load_chapters"):
                    try:
                        txt = pdf_text_from_path(path)
                        chapters = split_into_chapters(txt)
                        st.session_state._gen_book_chapters = chapters
                        st.success(f"Found {len(chapters)} chapters.")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

                chapters = st.session_state.get("_gen_book_chapters") or []
                if chapters:
                    chapter_titles = [t for (t, _) in chapters]
                    idx = st.selectbox(
                        "Choose chapter",
                        options=list(range(len(chapters))),
                        format_func=lambda i: chapter_titles[i],
                        key="gen_chapter_idx",
                    )

                    title, ch_text = chapters[idx]
                    ch_id = f"book:{book}:ch{idx+1:03d}"

                    with st.expander("Preview chapter"):
                        preview = ch_text[:1000].strip()
                        st.text(preview + ("…" if len(ch_text) > 1000 else ""))

                    if st.button("✨ Generate with AI", key="gen_do_ai"):
                        try:
                            limited = (ch_text[:max_chars] or "").strip()
                            if not limited:
                                st.error("Chapter text is empty.")
                            else:
                                cards = generate_flashcards_with_gemini(
                                    chapter_text=limited,
                                    n=int(n_cards),
                                    default_domain=chapter_default_domain,
                                    chapter_id=ch_id,
                                    source_name=f"Library: {book} | {title}",
                                )

                                if not cards:
                                    st.error("AI returned no usable flashcards. Try again or reduce/adjust your chapter text.")
                                else:
                                    st.session_state._gen_cards_preview = cards
                                    st.success(f"Generated {len(cards)} flashcards. Review below and save to DB.")
                                    st.rerun()
                        except Exception as e:
                            st.error(str(e))

                preview_cards = st.session_state.get("_gen_cards_preview") or []
                if preview_cards:
                    st.markdown("### Generated flashcards (preview)")
                    st.dataframe(
                        [{"front": c["front"], "back": c["back"], "domain": c["domain"]} for c in preview_cards],
                        use_container_width=True,
                    )

                    if st.button("💾 Save generated flashcards to database", key="gen_save_db"):
                        bulk_add_flashcards(preview_cards)
                        st.success(f"Saved {len(preview_cards)} flashcards to database.")
                        st.session_state._gen_cards_preview = []
                        st.rerun()
