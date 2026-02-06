# ui/labs.py
from __future__ import annotations

import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st

from config import CCNA_BLUEPRINT
from db.database import (
    add_lab_record,
    delete_lab_record,
    list_labs,
    update_lab_progress,
    update_lab_steps,
    set_lab_last_opened,
    get_questions,  # from your db, used for weak-domain recommendations
)

LABS_DIR = Path("data/labs")
LABS_DIR.mkdir(parents=True, exist_ok=True)

# Packet Tracer 9.0.0 common paths on Windows 11
PACKET_TRACER_CANDIDATES = [
    r"C:\Program Files\Cisco Packet Tracer\PacketTracer.exe",
    r"C:\Program Files\Cisco Packet Tracer 9.0\PacketTracer.exe",
    r"C:\Program Files\Cisco Packet Tracer 9.0.0\PacketTracer.exe",
    r"C:\Program Files (x86)\Cisco Packet Tracer\PacketTracer.exe",
]


def _safe_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^\w\-. ()\[\]]+", "_", name)
    return name[:180] if len(name) > 180 else name


def _find_packet_tracer() -> str | None:
    for p in PACKET_TRACER_CANDIDATES:
        if Path(p).exists():
            return p
    return None


def _launch_packet_tracer(pkt_path: Path) -> None:
    exe = _find_packet_tracer()
    if not exe:
        raise FileNotFoundError(
            "PacketTracer.exe not found.\n"
            "If Packet Tracer is installed somewhere else, add its path to PACKET_TRACER_CANDIDATES in ui/labs.py."
        )
    subprocess.Popen([exe, str(pkt_path)], close_fds=True)


def _domain_accuracy() -> Dict[str, Tuple[int, int, float]]:
    """
    Returns domain -> (seen_total, correct_total, accuracy_pct)
    """
    qs = get_questions() or []
    stats: Dict[str, Tuple[int, int]] = {}
    for q in qs:
        d = q.get("domain") or "Unknown"
        seen = int(q.get("seen", 0) or 0)
        cor = int(q.get("correct", 0) or 0)
        if d not in stats:
            stats[d] = (0, 0)
        s0, c0 = stats[d]
        stats[d] = (s0 + seen, c0 + cor)

    out: Dict[str, Tuple[int, int, float]] = {}
    for d, (s, c) in stats.items():
        acc = (c / s * 100.0) if s > 0 else 0.0
        out[d] = (s, c, round(acc, 1))
    return out


def _weak_domains(threshold_pct: float = 75.0, min_seen: int = 10) -> List[str]:
    acc = _domain_accuracy()
    weak = []
    for d, (seen, _cor, pct) in acc.items():
        if seen >= min_seen and pct < threshold_pct:
            weak.append(d)
    return weak


def show_labs() -> None:
    st.header("🧪 Packet Tracer Labs")

    # ---------------- Upload / Add ----------------
    st.subheader("Add labs to your library")
    uploaded = st.file_uploader(
        "Upload Packet Tracer labs (.pkt)",
        type=["pkt"],
        accept_multiple_files=True,
    )

    domains = list(CCNA_BLUEPRINT.keys())
    domain = st.selectbox("Tag domain", domains)
    blueprint = st.selectbox("Tag blueprint objective", CCNA_BLUEPRINT[domain])
    difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
    title_override = st.text_input("Optional title override (leave blank to use filename)")

    if st.button("Save Labs", type="primary"):
        if not uploaded:
            st.warning("Please select at least one .pkt file.")
            return

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for f in uploaded:
            fname = _safe_filename(f.name)
            dest = LABS_DIR / fname
            dest.write_bytes(f.getvalue())

            title = title_override.strip() if title_override.strip() else fname.rsplit(".", 1)[0]
            add_lab_record(
                filename=fname,
                title=title,
                domain=domain,
                blueprint=blueprint,
                difficulty=difficulty,
                added_at=now,
            )

        st.success("Labs saved to library.")
        st.rerun()

    st.markdown("---")

    # ---------------- Recommendations ----------------
    st.subheader("Recommended labs (weak domains)")
    weak = _weak_domains(threshold_pct=75.0, min_seen=10)
    stats = _domain_accuracy()

    labs = list_labs()
    if not labs:
        st.info("No labs yet. Upload a .pkt file above.")
        return

    if weak:
        recs = [lab for lab in labs if (lab.get("domain") in weak and (lab.get("status") or "In Progress") != "Completed")]
        if recs:
            for lab in recs[:10]:
                st.write(f"**{lab.get('title', lab['filename'])}** — {lab.get('domain')} ({lab.get('difficulty','')})")
                st.caption(lab.get("blueprint", ""))
        else:
            st.success("Nice — no recommended labs found (either completed or none tagged to weak domains).")
    else:
        st.info("No weak domains detected yet (need more answered questions).")

    st.markdown("---")

    # ---------------- Library List ----------------
    st.subheader("Labs Library")

    exe = _find_packet_tracer()
    if exe:
        st.caption(f"✅ Packet Tracer detected: {exe}")
    else:
        st.warning("⚠️ Packet Tracer not detected. Launch will fail until PacketTracer.exe is found.")

    # Filters
    colf1, colf2, colf3 = st.columns(3)
    with colf1:
        f_domain = st.selectbox("Filter domain", ["All"] + domains)
    with colf2:
        f_status = st.selectbox("Filter status", ["All", "In Progress", "Completed"])
    with colf3:
        f_diff = st.selectbox("Filter difficulty", ["All", "Easy", "Medium", "Hard"])

    def _matches(lab: Dict) -> bool:
        if f_domain != "All" and lab.get("domain") != f_domain:
            return False
        if f_status != "All" and (lab.get("status") or "In Progress") != f_status:
            return False
        if f_diff != "All" and lab.get("difficulty") != f_diff:
            return False
        return True

    shown = [lab for lab in labs if _matches(lab)]
    if not shown:
        st.info("No labs match your filters.")
        return

    # Render each lab
    for lab in shown:
        fname = lab["filename"]
        path = LABS_DIR / fname
        title = lab.get("title") or fname
        lab_domain = lab.get("domain") or ""
        lab_status = lab.get("status") or "In Progress"
        notes = lab.get("notes") or ""
        steps_md = lab.get("steps_md") or ""
        last_opened = lab.get("last_opened_at") or ""

        # Header
        st.markdown(f"### {title}")
        acc_line = ""
        if lab_domain in stats:
            seen, _cor, pct = stats[lab_domain]
            if seen > 0:
                acc_line = f" • Your accuracy in this domain: **{pct}%** (seen {seen})"

        st.caption(f"{lab_domain} • {lab.get('difficulty','')} • {lab.get('blueprint','')}{acc_line}")
        if last_opened:
            st.caption(f"Last opened: {last_opened}")

        # Actions row
        a1, a2, a3, a4 = st.columns([1, 1, 1, 1])

        with a1:
            if st.button("Launch", key=f"launch_{fname}"):
                try:
                    _launch_packet_tracer(path)
                    set_lab_last_opened(fname, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    st.success("Launching Packet Tracer…")
                except Exception as e:
                    st.error(str(e))

        with a2:
            if path.exists():
                st.download_button(
                    "Download .pkt",
                    data=path.read_bytes(),
                    file_name=fname,
                    mime="application/octet-stream",
                    key=f"dl_{fname}",
                )
            else:
                st.caption("File missing on disk.")

        with a3:
            new_status = st.selectbox(
                "Status",
                ["In Progress", "Completed"],
                index=0 if lab_status == "In Progress" else 1,
                key=f"status_{fname}",
            )

        with a4:
            if st.button("Delete", key=f"del_{fname}"):
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass
                delete_lab_record(fname)
                st.rerun()

        # Notes + Steps editors
        ncol, scol = st.columns(2)

        with ncol:
            new_notes = st.text_area("Notes", value=notes, height=140, key=f"notes_{fname}")
            if st.button("Save notes", key=f"save_notes_{fname}"):
                update_lab_progress(fname, new_status, new_notes)
                st.success("Saved.")
                st.rerun()

        with scol:
            with st.expander("Lab steps (markdown checklist)"):
                st.caption("Tip: Use `- [ ]` for tasks, `- [x]` for completed steps.")
                new_steps = st.text_area("Steps markdown", value=steps_md, height=220, key=f"steps_{fname}")
                if st.button("Save steps", key=f"save_steps_{fname}"):
                    update_lab_steps(fname, new_steps)
                    st.success("Steps saved.")
                    st.rerun()

            with st.expander("View steps"):
                if steps_md.strip():
                    st.markdown(steps_md)
                else:
                    st.info("No steps yet. Add a checklist in the editor above.")

        # Persist status even if they only changed dropdown
        # (so status doesn’t “look changed” but not saved)
        # Save status change automatically only if notes unchanged:
        if st.button("Save status", key=f"save_status_{fname}"):
            update_lab_progress(fname, new_status, new_notes)
            st.success("Status saved.")
            st.rerun()

        st.markdown("---")
