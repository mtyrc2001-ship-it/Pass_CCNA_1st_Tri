# ui/exam.py
from __future__ import annotations

import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, Any, List

from db.database import (
    update_question_progress,
    update_coverage,
    add_history,
    get_questions,
)
from config import DOMAIN_WEIGHTS, DIFFICULTY_WEIGHT
from engine.questions import select_adaptive_questions


def confidence_band(score: float) -> str:
    if score >= 90:
        return "Very High"
    if score >= 80:
        return "High"
    if score >= 70:
        return "Borderline"
    return "Low"


def readiness_score(weighted: float, weak_domains: List[str], avg_diff: float, blueprint_gaps: int) -> float:
    penalty = len(weak_domains) * 5 + blueprint_gaps * 3
    bonus = avg_diff * 5
    return max(0.0, min(100.0, weighted - penalty + bonus))


def _init_exam_state() -> None:
    """Initialize all exam-specific session_state keys safely."""
    if "exam_questions" not in st.session_state:
        st.session_state.exam_questions = []
    if "exam_idx" not in st.session_state:
        st.session_state.exam_idx = 0
    if "exam_answers" not in st.session_state:
        st.session_state.exam_answers = {}  # idx -> selected answer
    if "exam_done" not in st.session_state:
        st.session_state.exam_done = False
    if "exam_end_time" not in st.session_state:
        st.session_state.exam_end_time = None
    if "exam_scored" not in st.session_state:
        # prevents double-updating DB on reruns after completion
        st.session_state.exam_scored = False


def _start_exam(domain: str, n_questions: int, seconds_per_question: int) -> None:
    qs = select_adaptive_questions(domain, n_questions) or []
    st.session_state.exam_questions = qs
    st.session_state.exam_idx = 0
    st.session_state.exam_answers = {}
    st.session_state.exam_done = False
    st.session_state.exam_scored = False
    st.session_state.exam_end_time = datetime.now() + timedelta(seconds=len(qs) * seconds_per_question)


def show_exam() -> None:
    _init_exam_state()

    st.header("🧪 Exam Simulation")

    # ---- Controls ----
    colA, colB, colC = st.columns([2, 1, 1])
    with colA:
        domain = st.selectbox("Domain Focus", list(DOMAIN_WEIGHTS.keys()))
    with colB:
        n_questions = st.number_input("Questions", min_value=5, max_value=60, value=20, step=5)
    with colC:
        seconds_per_question = st.number_input("Sec / Q", min_value=30, max_value=180, value=75, step=5)

    c1, c2 = st.columns([1, 1])
    with c1:
        start_clicked = st.button("Start / Restart Exam", type="primary")
    with c2:
        reset_clicked = st.button("Reset State")

    if reset_clicked:
        st.session_state.exam_questions = []
        st.session_state.exam_idx = 0
        st.session_state.exam_answers = {}
        st.session_state.exam_done = False
        st.session_state.exam_scored = False
        st.session_state.exam_end_time = None
        st.rerun()

    if start_clicked:
        _start_exam(domain, int(n_questions), int(seconds_per_question))
        st.rerun()

    # Auto-start if no exam yet (optional)
    if not st.session_state.exam_questions:
        st.info("Click **Start / Restart Exam** to begin.")
        return

    # ---- Timer ----
    if st.session_state.exam_end_time is not None and not st.session_state.exam_done:
        rem = st.session_state.exam_end_time - datetime.now()
        if rem.total_seconds() <= 0:
            st.session_state.exam_done = True
            st.rerun()
        minutes, seconds = divmod(int(rem.total_seconds()), 60)
        st.metric("Time Left", f"{minutes:02d}:{seconds:02d}")

    # Clamp idx
    if st.session_state.exam_idx < 0:
        st.session_state.exam_idx = 0
    if st.session_state.exam_idx > len(st.session_state.exam_questions) - 1:
        st.session_state.exam_idx = len(st.session_state.exam_questions) - 1

    # ---- Exam flow ----
    if not st.session_state.exam_done:
        q: Dict[str, Any] = st.session_state.exam_questions[st.session_state.exam_idx]

        q_text = q.get("question", "")
        q_diff = q.get("difficulty", "Medium")
        q_opts = q.get("options", [])
        q_bp = q.get("blueprint", "")
        q_domain = q.get("domain", "Unknown")

        st.subheader(f"[{q_diff}] {q_text}")
        st.caption(f"Domain: {q_domain} • Blueprint: {q_bp}" if q_bp else f"Domain: {q_domain}")

        if not q_opts:
            st.error("This question has no options. Check your question generator output.")
            return

        # Default selection if already answered
        prev = st.session_state.exam_answers.get(st.session_state.exam_idx, q_opts[0])
        try:
            prev_index = q_opts.index(prev)
        except ValueError:
            prev_index = 0

        selected = st.radio(
            "Choose",
            q_opts,
            index=prev_index,
            key=f"exam_q_{st.session_state.exam_idx}",
        )
        st.session_state.exam_answers[st.session_state.exam_idx] = selected

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("Previous", disabled=(st.session_state.exam_idx <= 0)):
                st.session_state.exam_idx -= 1
                st.rerun()
        with col2:
            if st.button("Submit & Next"):
                if st.session_state.exam_idx < len(st.session_state.exam_questions) - 1:
                    st.session_state.exam_idx += 1
                else:
                    st.session_state.exam_done = True
                st.rerun()
        with col3:
            if st.button("Finish Exam Now"):
                st.session_state.exam_done = True
                st.rerun()

    # ---- Results ----
    else:
        # Prevent re-scoring on reruns
        if not st.session_state.exam_scored:
            questions = st.session_state.exam_questions
            answers = st.session_state.exam_answers

            domain_stats = {d: {"c": 0, "t": 0} for d in DOMAIN_WEIGHTS}
            diff_sum = 0.0

            for idx, q in enumerate(questions):
                d = q.get("domain")
                if d not in domain_stats:
                    # ignore unknown domains
                    continue

                domain_stats[d]["t"] += 1

                diff = q.get("difficulty", "Medium")
                diff_sum += float(DIFFICULTY_WEIGHT.get(diff, 1.0))

                correct = (answers.get(idx) == q.get("answer"))

                if correct:
                    domain_stats[d]["c"] += 1

                # Update DB only if ID present
                qid = q.get("id")
                if qid:
                    update_question_progress(qid, bool(correct))

                bp = q.get("blueprint")
                if bp:
                    update_coverage(bp, bool(correct))

            # Weighted score across attempted domains
            weighted = 0.0
            for d, v in domain_stats.items():
                if v["t"] > 0:
                    weighted += (v["c"] / v["t"]) * DOMAIN_WEIGHTS.get(d, 0) * 100.0

            weak = [
                d
                for d, v in domain_stats.items()
                if v["t"] > 0 and (v["c"] / v["t"]) * 100.0 < 75.0
            ]

            # Blueprint gaps (using question-level stats, as your original did)
            try:
                qrows = get_questions()
                blueprint_gaps = sum(
                    1
                    for r in qrows
                    if (r.get("seen", 0) or 0) > 0 and ((r.get("correct", 0) or 0) / (r.get("seen", 1) or 1)) < 0.7
                )
            except Exception:
                blueprint_gaps = 0

            avg_diff = (diff_sum / len(questions)) if questions else 0.0
            readiness = readiness_score(weighted, weak, avg_diff, blueprint_gaps)

            st.session_state.exam_report = {
                "weighted": weighted,
                "band": confidence_band(weighted),
                "readiness": readiness,
            }

            add_history(datetime.now().strftime("%Y-%m-%d"), weighted, readiness, confidence_band(weighted))

            st.session_state.exam_scored = True

        report = st.session_state.get("exam_report", {})
        weighted = float(report.get("weighted", 0.0))
        band = report.get("band", confidence_band(weighted))
        readiness = float(report.get("readiness", 0.0))

        st.header("📊 Boson-Style Final Report")
        st.metric("Final Score", f"{weighted:.1f}%")
        st.metric("Confidence Band", band)
        st.metric("Readiness Score", f"{readiness:.0f}/100")

        st.markdown("---")
        if st.button("Start Another Exam"):
            # restart with current selections
            _start_exam(domain, int(n_questions), int(seconds_per_question))
            st.rerun()
