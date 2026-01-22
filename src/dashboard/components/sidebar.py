"""Sidebar component for navigation and work selection."""

import streamlit as st
from typing import Optional, List, Dict, Any


def render_sidebar(books: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Render the sidebar with work selection and navigation.

    Returns:
        Dictionary with selected book info and view options
    """
    st.sidebar.title("📚 Kafka Analysis")
    st.sidebar.markdown("---")

    # Work selector
    st.sidebar.subheader("Select Work")
    st.sidebar.caption("Choose a work for detailed analysis pages")

    book_options = {book["title"]: book["id"] for book in books}
    book_titles = list(book_options.keys())

    if not book_titles:
        st.sidebar.warning("No books available")
        return {"book_id": None, "comparison_mode": False}

    selected_title = st.sidebar.selectbox(
        "Choose a work to analyze:",
        book_titles,
        key="book_selector"
    )

    selected_book_id = book_options[selected_title]
    selected_book = next((b for b in books if b["id"] == selected_book_id), None)

    # Display selected book info
    if selected_book:
        st.sidebar.markdown(f"""
        **Author:** {selected_book['author']}
        **Words:** {selected_book['word_count']:,}
        """)

    st.sidebar.markdown("---")

    # Comparison mode toggle
    st.sidebar.subheader("Options")
    comparison_mode = st.sidebar.checkbox(
        "Enable Comparison Mode",
        key="comparison_mode",
        help="Compare metrics across both works"
    )

    # If comparison mode, select second book
    second_book_id = None
    if comparison_mode and len(books) > 1:
        other_titles = [t for t in book_titles if t != selected_title]
        second_title = st.sidebar.selectbox(
            "Compare with:",
            other_titles,
            key="second_book_selector"
        )
        second_book_id = book_options[second_title]

    st.sidebar.markdown("---")

    # About section
    with st.sidebar.expander("ℹ️ About", expanded=False):
        st.markdown("""
        **Literary Analysis Dashboard**

        This dashboard visualizes NLP analysis results
        for Franz Kafka's English works:
        - The Metamorphosis
        - The Trial

        **Analysis Types:**
        - Sentiment & Emotional Arc
        - Readability & Complexity
        - Theme Extraction
        - Character Networks

        *Phase 3 of Literary Analysis Project*
        """)

    return {
        "book_id": selected_book_id,
        "book_title": selected_title,
        "book": selected_book,
        "comparison_mode": comparison_mode,
        "second_book_id": second_book_id,
    }
