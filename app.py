# app.py
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# ---- Ensure imports work no matter where Streamlit is launched from ----
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import APP_TITLE, APP_ICON  # noqa: E402
from db.database import init_db  # noqa: E402
from ui import dashboard, practice, exam, flashcards, library, labs  # noqa: E402


st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")
init_db()

# ---- Session state defaults (shared keys used across pages) ----
defaults = {
    # Navigation
    "mode": "Dashboard",
    # Practice page (legacy/shared)
    "questions": [],
    "idx": 0,
    "checked_ids": set(),
    "last_selected": {},
    # (Exam uses its own exam_* keys in ui/exam.py)
    # Flashcards uses its own fc_* keys in ui/flashcards.py
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.sidebar.title("Rudy")
st.sidebar.title("🛡️ CCNA Mastery Hub")

st.session_state.mode = st.sidebar.selectbox(
    "Mode",
    ["Dashboard", "Practice Mode", "Exam Simulation", "Flashcards", "Books Library" , "Labs"]

)

if st.session_state.mode == "Dashboard":
    dashboard.show_dashboard()
elif st.session_state.mode == "Practice Mode":
    practice.show_practice()
elif st.session_state.mode == "Exam Simulation":
    exam.show_exam()
elif st.session_state.mode == "Flashcards":
    flashcards.show_flashcards()
elif st.session_state.mode == "Books Library":
    library.show_library()
elif st.session_state.mode == "Labs":
    labs.show_labs()



