# ui/library.py
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st

from config import CCNA_BLUEPRINT
from db.database import add_book_record, delete_book_record, list_books


BOOKS_DIR = Path("data/books")
BOOKS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^\w\-. ()\[\]]+", "_", name)
    return name[:180] if len(name) > 180 else name


def pdf_text_and_pages(file_path: Path) -> Tuple[str, int]:
    try:
        import fitz  # PyMuPDF
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Missing dependency: PyMuPDF. Install with: pip install pymupdf") from e

    doc = fitz.open(str(file_path))
    try:
        text = "".join(page.get_text() for page in doc)
        return text, doc.page_count
    finally:
        doc.close()


def show_library() -> None:
    st.header("📚 Books Library")

    st.subheader("Upload CCNA PDFs (stored locally in data/books)")
    uploaded = st.file_uploader(
        "Upload one or more PDFs",
        type="pdf",
        accept_multiple_files=True,
    )

    if st.button("Save to Library", type="primary"):
        if not uploaded:
            st.warning("Please choose at least one PDF.")
            return

        for f in uploaded:
            fname = _safe_filename(f.name)
            dest = BOOKS_DIR / fname
            dest.write_bytes(f.getvalue())

            # best-effort pages count
            pages = 0
            try:
                _, pages = pdf_text_and_pages(dest)
            except Exception:
                pages = 0

            add_book_record(
                filename=fname,
                title=fname.rsplit(".", 1)[0],
                added_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                pages=pages,
            )

        st.success("Saved to library.")
        st.rerun()

    st.markdown("---")
    st.subheader("Your Library")

    books = list_books()
    if not books:
        st.info("No books saved yet.")
        return

    for b in books:
        fname = b["filename"]
        title = b.get("title", fname)
        pages = b.get("pages", 0)
        st.write(f"**{title}** — `{fname}`" + (f" • {pages} pages" if pages else ""))

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Delete", key=f"del_{fname}"):
                # delete file + record
                try:
                    (BOOKS_DIR / fname).unlink(missing_ok=True)
                except Exception:
                    pass
                delete_book_record(fname)
                st.rerun()

        with col2:
            # Streamlit can offer download for saved file
            try:
                data = (BOOKS_DIR / fname).read_bytes()
                st.download_button(
                    "Download PDF",
                    data=data,
                    file_name=fname,
                    mime="application/pdf",
                    key=f"dl_{fname}",
                )
            except Exception:
                st.caption("Could not load file for download.")
