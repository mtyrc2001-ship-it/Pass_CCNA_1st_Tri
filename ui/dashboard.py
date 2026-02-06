# ui/dashboard.py
from __future__ import annotations

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from typing import Dict

from db.database import get_coverage, get_history, get_questions, reset_history
from config import DOMAIN_WEIGHTS, CCNA_BLUEPRINT


# ---------------- Utility functions ----------------
def confidence_band(score: float) -> str:
    if score >= 90:
        return "Very High"
    if score >= 80:
        return "High"
    if score >= 70:
        return "Borderline"
    return "Low"


def calculate_weighted_domain_score(domain_stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for d, v in domain_stats.items():
        seen = float(v.get("seen", 0) or 0)
        correct = float(v.get("correct", 0) or 0)
        if seen > 0:
            acc = correct / seen
            scores[d] = acc * DOMAIN_WEIGHTS.get(d, 0) * 100
        else:
            scores[d] = 0.0
    return scores


def _safe_date_range(min_dt: pd.Timestamp, max_dt: pd.Timestamp):
    picked = st.sidebar.date_input(
        "Select Date Range",
        [min_dt.date(), max_dt.date()],
    )

    if isinstance(picked, date):
        return pd.to_datetime(picked), pd.to_datetime(picked)

    if isinstance(picked, (list, tuple)) and len(picked) == 2:
        return pd.to_datetime(picked[0]), pd.to_datetime(picked[1])

    return min_dt, max_dt


# ---------------- Main dashboard ----------------
def show_dashboard() -> None:
    st.title("📊 CCNA Mastery Interactive Dashboard")

    # ---------- Filters ----------
    st.sidebar.header("📌 Filters")

    history = get_history()
    if history:
        df_hist = pd.DataFrame(history)

        if "date" in df_hist.columns:
            df_hist["date"] = pd.to_datetime(df_hist["date"], errors="coerce")
            df_hist = df_hist.dropna(subset=["date"])

            if not df_hist.empty:
                start_date, end_date = _safe_date_range(
                    df_hist["date"].min(),
                    df_hist["date"].max(),
                )
                df_hist = df_hist[(df_hist["date"] >= start_date) & (df_hist["date"] <= end_date)]
    else:
        df_hist = pd.DataFrame()

    domains = st.sidebar.multiselect(
        "Filter Domains",
        list(DOMAIN_WEIGHTS.keys()),
        default=list(DOMAIN_WEIGHTS.keys()),
    )

    all_blueprints: list[str] = []
    for d in domains:
        all_blueprints.extend(CCNA_BLUEPRINT.get(d, []))

    blueprints = st.sidebar.multiselect(
        "Filter Blueprint Objectives",
        all_blueprints,
        default=all_blueprints,
    )

    st.markdown("---")

    # ---------- Readiness Trend ----------
    st.subheader("📈 Readiness Over Time")

    # Reset button for readiness/history only
    with st.expander("⚠️ Reset Readiness Trend", expanded=False):
        st.warning(
            "This will permanently delete your readiness history used in this chart.\n\n"
            "It will NOT delete questions, blueprint coverage, books, or labs."
        )
        confirm = st.checkbox("I understand this cannot be undone", key="confirm_reset_history")
        if st.button("🗑️ Clear Readiness History", disabled=not confirm, key="btn_reset_history"):
            reset_history()
            st.success("Readiness history cleared.")
            st.rerun()

    if not df_hist.empty and {"date", "readiness"}.issubset(df_hist.columns):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df_hist["date"], df_hist["readiness"], marker="o")
        ax.set_xlabel("Date")
        ax.set_ylabel("Readiness Score")
        ax.grid(True)
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("No readiness data available.")

    st.markdown("---")

    # ---------- Blueprint Coverage ----------
    st.subheader("📘 Blueprint Coverage")
    coverage = get_coverage()

    if coverage:
        df_cov = pd.DataFrame.from_dict(coverage, orient="index")
        df_cov.index.name = "Blueprint Objective"

        if blueprints:
            df_cov = df_cov[df_cov.index.isin(blueprints)]

        if not df_cov.empty and {"seen", "correct"}.issubset(df_cov.columns):
            df_cov["seen"] = pd.to_numeric(df_cov["seen"], errors="coerce").fillna(0)
            df_cov["correct"] = pd.to_numeric(df_cov["correct"], errors="coerce").fillna(0)

            df_cov["accuracy"] = (
                df_cov["correct"] / df_cov["seen"].where(df_cov["seen"] > 0, 1)
            ).round(2)

            st.dataframe(df_cov, use_container_width=True)

            st.subheader("⚠️ Top 5 Weakest Blueprint Objectives")
            weak = df_cov[df_cov["seen"] > 0].sort_values("accuracy").head(5)
            if not weak.empty:
                st.table(weak[["accuracy"]])
            else:
                st.info("No attempted blueprint objectives yet.")
        else:
            st.info("No blueprint coverage for selected objectives.")
    else:
        st.info("No blueprint coverage yet.")

    st.markdown("---")

    # ---------- Domain Weaknesses ----------
    st.subheader("🛠️ Domain Weaknesses")
    questions = get_questions() or []

    if not domains:
        domains = list(DOMAIN_WEIGHTS.keys())

    domain_stats: Dict[str, Dict[str, int]] = {d: {"seen": 0, "correct": 0} for d in domains}

    for q in questions:
        d = q.get("domain")
        if d in domain_stats:
            domain_stats[d]["seen"] += int(q.get("seen", 0) or 0)
            domain_stats[d]["correct"] += int(q.get("correct", 0) or 0)

    rows = []
    for d, v in domain_stats.items():
        seen = int(v.get("seen", 0) or 0)
        cor = int(v.get("correct", 0) or 0)
        acc_pct = round((cor / seen * 100) if seen > 0 else 0.0, 1)

        rows.append(
            {
                "Domain": d,
                "Questions Seen": seen,
                "Correct": cor,
                "Accuracy (%)": acc_pct,
                "Status": "Weak" if seen > 0 and acc_pct < 75 else "Strong",
            }
        )

    df_dom = pd.DataFrame(rows)

    if df_dom.empty or "Domain" not in df_dom.columns:
        st.info("No domain performance data yet. Answer some questions in Practice/Exam mode first.")
        return

    weighted_scores = calculate_weighted_domain_score(domain_stats)
    df_dom["Weighted Score"] = df_dom["Domain"].map(weighted_scores).fillna(0.0)
    df_dom["Confidence Band"] = df_dom["Accuracy (%)"].apply(confidence_band)

    st.dataframe(df_dom, use_container_width=True)

    # Optional heatmap
    try:
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(8, max(2, len(df_dom) * 0.45)))
        pivot = df_dom.pivot(index="Domain", columns="Status", values="Accuracy (%)").fillna(0)
        sns.heatmap(pivot, annot=True, cmap="coolwarm", cbar=True, ax=ax)
        ax.set_title("Domain Accuracy Heatmap")
        st.pyplot(fig, clear_figure=True)
    except Exception:
        st.info("Install seaborn for heatmap visualization: pip install seaborn")
