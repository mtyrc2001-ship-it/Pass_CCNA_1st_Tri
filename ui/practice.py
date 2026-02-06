# ui/practice.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from config import CCNA_BLUEPRINT
from engine.questions import gen_questions, select_adaptive_questions
from db.database import list_books, update_coverage, update_question_progress

BOOKS_DIR = Path("data/books")
DOMAINS = list(CCNA_BLUEPRINT.keys())

# ---------------------------
# Domain keyword heuristics (used to suggest a domain per chapter)
# ---------------------------
DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "Network Fundamentals": ["osi", "tcp", "udp", "ethernet", "cabling", "subnet", "ipv4", "ipv6", "switch", "router"],
    "Network Access": ["vlan", "trunk", "stp", "spanning", "etherchannel", "portfast", "wireless", "wlc", "capwap"],
    "IP Connectivity": ["routing", "static route", "ospf", "default route", "next hop", "routing table"],
    "IP Services": ["dhcp", "dns", "ntp", "snmp", "syslog", "nat", "qos"],
    "Security Fundamentals": ["acl", "aaa", "port security", "vpn", "security", "wpa", "802.1x"],
    "Automation and Programmability": ["api", "rest", "json", "yaml", "automation", "controller", "netconf", "restconf"],
    "Wireless": ["wlc", "capwap", "ssid", "802.11", "roaming", "rf", "channels"],
}


# ---------------------------
# Session init
# ---------------------------
def _init_state() -> None:
    # practice quiz state
    st.session_state.setdefault("questions", [])
    st.session_state.setdefault("idx", 0)
    st.session_state.setdefault("checked_ids", set())
    st.session_state.setdefault("last_selected", {})

    # sources/chapter cache
    st.session_state.setdefault("src_loaded", False)
    st.session_state.setdefault("src_text", "")
    st.session_state.setdefault("src_chapters", [])  # list[dict]
    st.session_state.setdefault("src_sources", [])

    # UI selections
    st.session_state.setdefault("pm_mode", "Single chapter")
    st.session_state.setdefault("pm_domain", DOMAINS[0] if DOMAINS else "Network Fundamentals")
    st.session_state.setdefault("pm_chapter_id", "(none)")
    st.session_state.setdefault("pm_override_domain", "Auto")
    st.session_state.setdefault("pm_qn_count", 20)
    st.session_state.setdefault("pm_preview_len", 900)


# ---------------------------
# PDF extraction
# ---------------------------
def _require_pymupdf():
    try:
        import fitz  # PyMuPDF
        return fitz
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Missing dependency: PyMuPDF. Install with: pip install pymupdf") from e


def pdf_text_from_upload(uploaded_file) -> str:
    fitz = _require_pymupdf()
    data = uploaded_file.read()
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        return "".join(page.get_text() for page in doc)
    finally:
        doc.close()


def pdf_text_from_path(path: Path) -> str:
    fitz = _require_pymupdf()
    doc = fitz.open(str(path))
    try:
        return "".join(page.get_text() for page in doc)
    finally:
        doc.close()


def _collect_source_texts(selected_library_files: List[str], uploaded_files) -> Tuple[str, List[str]]:
    combined_parts: List[str] = []
    sources: List[str] = []

    # Library PDFs
    for fname in selected_library_files:
        path = BOOKS_DIR / fname
        if not path.exists():
            continue
        combined_parts.append(f"\n\n===== LIBRARY BOOK: {fname} =====\n\n")
        combined_parts.append(pdf_text_from_path(path))
        sources.append(f"Library: {fname}")

    # Uploaded PDFs
    for f in (uploaded_files or []):
        combined_parts.append(f"\n\n===== UPLOAD: {f.name} =====\n\n")
        combined_parts.append(pdf_text_from_upload(f))
        sources.append(f"Upload: {f.name}")

    return "\n".join(combined_parts), sources


# ---------------------------
# Chapter splitting + tagging
# ---------------------------
def split_into_chapters(text: str) -> List[Tuple[str, str]]:
    """
    Heuristic: split on lines like 'Chapter 1 ...' or 'Unit 2 ...'.
    If none found, returns a single "Full Document" chapter.
    """
    pattern = re.compile(r"(?im)^\s*(chapter|unit)\s+\d+.*$", re.MULTILINE)
    matches = list(pattern.finditer(text or ""))

    if not matches:
        return [("Full Document", text or "")]

    chunks: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        start = m.start()
        # ✅ FIX: compare against number of matches, not length of text
        end = matches[i + 1].start() if (i + 1) < len(matches) else len(text)
        title_line = (m.group(0) or "").strip()
        chunk_text = (text or "")[start:end]
        chunks.append((title_line, chunk_text))

    return chunks


def _score_domain(chunk_text: str, domain: str) -> float:
    t = (chunk_text or "").lower()
    kws = DOMAIN_KEYWORDS.get(domain, [])
    if not kws:
        return 0.0
    hits = sum(1 for kw in kws if kw in t)
    return hits / len(kws)


def suggest_domain_for_chapter(chapter_text: str) -> str:
    scores = [(d, _score_domain(chapter_text, d)) for d in DOMAINS]
    best_domain, best_score = max(scores, key=lambda x: x[1])
    return best_domain if best_score > 0 else "Network Fundamentals"


def build_chapter_objects(text: str) -> List[Dict[str, Any]]:
    chapters = split_into_chapters(text or "")
    out: List[Dict[str, Any]] = []
    for i, (title, ctext) in enumerate(chapters):
        ch_id = f"ch_{i+1:03d}"
        out.append(
            {
                "id": ch_id,
                "title": title,
                "text": ctext,
                "suggested_domain": suggest_domain_for_chapter(ctext),
                "length": len(ctext or ""),
            }
        )
    return out


def _get_chapter_by_id(chapters: List[Dict[str, Any]], ch_id: str) -> Optional[Dict[str, Any]]:
    for c in chapters:
        if c.get("id") == ch_id:
            return c
    return None


def _effective_domain_for_selected_chapter(chapter: Dict[str, Any], override: str) -> str:
    if override and override != "Auto" and override in DOMAINS:
        return override
    return chapter.get("suggested_domain") or "Network Fundamentals"


# ---------------------------
# UI
# ---------------------------
def show_practice() -> None:
    _init_state()

    st.header("🧠 Practice Mode")

    # --------------------------
    # 1) Choose study sources
    # --------------------------
    st.subheader("1) Choose study sources")

    books = list_books() or []
    library_filenames = [b.get("filename") for b in books if b.get("filename")]

    colL, colU = st.columns(2)

    with colL:
        use_library = st.checkbox("Use Books Library", value=True)
        selected_library = st.multiselect(
            "Select books from Library",
            options=library_filenames,
            default=[],
            disabled=(not use_library),
        )
        if use_library and not library_filenames:
            st.info("No library books found. Add PDFs in the **Books Library** tab first.")

    with colU:
        use_uploads = st.checkbox("Upload new PDFs", value=True)
        uploads = st.file_uploader(
            "Upload PDFs (multi-upload)",
            type="pdf",
            accept_multiple_files=True,
            disabled=(not use_uploads),
        )

    colA, colB = st.columns([1, 3])
    with colA:
        load_sources = st.button("Load Sources & Extract Chapters", type="primary")
    with colB:
        st.caption("Extracts text, splits into chapters, and suggests a domain per chapter.")

    if load_sources:
        if (not use_library or not selected_library) and (not use_uploads or not uploads):
            st.warning("Select at least one library book and/or upload at least one PDF.")
            return

        if use_library and selected_library and not BOOKS_DIR.exists():
            st.error("Books folder not found. Open the Books Library tab once to create it.")
            return

        try:
            full_text, sources = _collect_source_texts(
                selected_library_files=(selected_library if use_library else []),
                uploaded_files=(uploads if use_uploads else []),
            )
        except Exception as e:
            st.error(str(e))
            return

        if not (full_text or "").strip():
            st.error("No text extracted from selected sources. (Are the PDFs scanned images?)")
            st.info("Image-only PDFs need OCR (not implemented yet).")
            return

        chapters = build_chapter_objects(full_text)

        st.session_state.src_text = full_text
        st.session_state.src_sources = sources
        st.session_state.src_chapters = chapters
        st.session_state.src_loaded = True

        # reset selections
        st.session_state.pm_chapter_id = chapters[0]["id"] if chapters else "(none)"
        st.session_state.pm_override_domain = "Auto"

        st.success(f"Loaded {len(chapters)} chapters from {len(sources)} source(s).")
        st.rerun()

    st.markdown("---")

    # --------------------------
    # 2) Generate (top)
    # --------------------------
    st.subheader("2) Generate (top)")

    if not st.session_state.src_loaded:
        st.info("Load sources first to generate questions.")
        return

    chapters = st.session_state.src_chapters or []
    if not chapters:
        st.error("No chapters were detected.")
        return

    if st.session_state.src_sources:
        st.caption("Sources: " + ", ".join(st.session_state.src_sources))

    bar1, bar2, bar3, bar4 = st.columns([2, 2, 3, 2])

    with bar1:
        st.selectbox("Mode", ["Single chapter", "Domain only"], key="pm_mode")

    with bar2:
        st.slider("Total questions", 5, 50, int(st.session_state.pm_qn_count), key="pm_qn_count")

    with bar3:
        if st.session_state.pm_mode == "Domain only":
            st.selectbox("Domain", DOMAINS, key="pm_domain")
        else:
            chapter_ids = [c["id"] for c in chapters]
            labels = {c["id"]: (c.get("title") or c["id"]) for c in chapters}
            st.selectbox(
                "Chapter",
                options=chapter_ids,
                key="pm_chapter_id",
                format_func=lambda cid: f"{cid} — {labels.get(cid, cid)}",
            )

    with bar4:
        if st.session_state.pm_mode == "Single chapter":
            st.selectbox("Override", ["Auto"] + DOMAINS, key="pm_override_domain")
        else:
            st.caption("")

    if st.session_state.pm_mode == "Single chapter":
        sel = _get_chapter_by_id(chapters, st.session_state.pm_chapter_id)
        if sel:
            with st.expander("Preview selected chapter", expanded=False):
                st.slider("Preview length (chars)", 500, 2000, int(st.session_state.pm_preview_len), step=100, key="pm_preview_len")
                txt = (sel.get("text") or "").strip()
                pl = int(st.session_state.pm_preview_len)
                st.text(txt[:pl] + ("…" if len(txt) > pl else ""))

    gen_clicked = st.button("Generate Questions", type="primary")

    # moved below button
    if not st.session_state.questions and not gen_clicked:
        st.info("Generate questions to start practicing.")

    if gen_clicked:
        try:
            n = int(st.session_state.pm_qn_count)

            if st.session_state.pm_mode == "Domain only":
                domain = st.session_state.pm_domain
                text_for_domain = (st.session_state.src_text or "").strip()
                if not text_for_domain:
                    st.warning("No extracted text available.")
                    return

                gen_questions(text_for_domain, n, domain)
                qs = select_adaptive_questions(domain, n) or []
                st.session_state.questions = qs[:n]

            else:
                ch_id = st.session_state.pm_chapter_id
                chapter = _get_chapter_by_id(chapters, ch_id)
                if not chapter:
                    st.warning("Selected chapter not found.")
                    return

                eff_domain = _effective_domain_for_selected_chapter(chapter, st.session_state.pm_override_domain)
                ch_text = (chapter.get("text") or "").strip()
                if not ch_text:
                    st.warning("Selected chapter has no extracted text.")
                    return

                gen_questions(
                    ch_text,
                    n,
                    eff_domain,
                    chapter_id=ch_id,
                    chapter_title=chapter.get("title"),
                    source_name="; ".join(st.session_state.src_sources or [])[:500],
                )

                qs = select_adaptive_questions(eff_domain, n, chapter_ids=[ch_id]) or []
                st.session_state.questions = qs[:n]

            st.session_state.idx = 0
            st.session_state.checked_ids = set()
            st.session_state.last_selected = {}
            st.rerun()

        except Exception as e:
            st.error(str(e))
            return

    st.markdown("---")

    # --------------------------
    # Quiz (Practice Questions)
    # --------------------------
    if not st.session_state.questions:
        return

    st.session_state.idx = max(0, min(int(st.session_state.idx), len(st.session_state.questions) - 1))
    q = st.session_state.questions[st.session_state.idx]

    qid = q.get("id", f"idx-{st.session_state.idx}")
    difficulty = q.get("difficulty", "Medium")
    question_text = q.get("question", "")
    blueprint = q.get("blueprint", "")
    options = q.get("options", [])
    answer = q.get("answer", None)
    explanation = q.get("explanation", "")

    st.subheader(f"[{difficulty}] {question_text}")
    if blueprint:
        st.caption(f"Blueprint: {blueprint}")

    if not options:
        st.error("Question has no options. Check your generator output.")
        return

    default_choice = st.session_state.last_selected.get(qid, options[0])
    try:
        default_index = options.index(default_choice)
    except ValueError:
        default_index = 0

    selected_answer = st.radio("Answer", options, index=default_index, key=f"ans_{qid}")
    st.session_state.last_selected[qid] = selected_answer

    already_checked = qid in st.session_state.checked_ids
    grade_clicked = st.button("Check Answer", disabled=already_checked)

    if grade_clicked and not already_checked:
        is_correct = (selected_answer == answer)

        try:
            if q.get("id"):
                update_question_progress(q["id"], bool(is_correct))
            if blueprint:
                update_coverage(blueprint, bool(is_correct))
        except Exception as e:
            st.error(f"Could not update database: {e}")
            return

        st.session_state.checked_ids.add(qid)

        if is_correct:
            st.success("Correct ✔️")
        else:
            st.error("Incorrect ❌")
            if explanation:
                st.info(explanation)

    nav1, nav2 = st.columns(2)
    with nav1:
        if st.button("Previous", disabled=(st.session_state.idx <= 0)):
            st.session_state.idx -= 1
            st.rerun()
    with nav2:
        if st.button("Next", disabled=(st.session_state.idx >= len(st.session_state.questions) - 1)):
            st.session_state.idx += 1
            st.rerun()
