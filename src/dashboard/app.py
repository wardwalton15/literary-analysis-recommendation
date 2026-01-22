"""Main Streamlit dashboard application."""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.dashboard.utils.data_loader import get_kafka_books
from src.dashboard.components.sidebar import render_sidebar

# Page configuration
st.set_page_config(
    page_title="Kafka Literary Analysis",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application entry point."""
    # Load books data
    books = get_kafka_books()

    if not books:
        st.error("No Kafka books found in the database. Please run the analysis pipeline first.")
        st.info("Run: `python -m src.cli analyze --kafka` to analyze Kafka's works.")
        return

    # Render sidebar and get selections
    sidebar_state = render_sidebar(books)

    # Store selections in session state
    if "book_id" not in st.session_state:
        st.session_state.book_id = sidebar_state["book_id"]

    # Navigation
    pages = {
        "🏠 Home": "home",
        "💭 Sentiment Analysis": "sentiment",
        "📊 Complexity Metrics": "complexity",
        "🎯 Themes & Topics": "topics",
        "🔗 Character Network": "characters",
        "⚖️ Comparison": "comparison",
    }

    # Tab-based navigation
    selected_page = st.radio(
        "Navigate",
        list(pages.keys()),
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Route to appropriate page
    page_key = pages[selected_page]

    if page_key == "home":
        from src.dashboard.pages.home import render_home_page
        render_home_page(books, sidebar_state)

    elif page_key == "sentiment":
        from src.dashboard.pages.sentiment import render_sentiment_page
        render_sentiment_page(sidebar_state)

    elif page_key == "complexity":
        from src.dashboard.pages.complexity import render_complexity_page
        render_complexity_page(sidebar_state)

    elif page_key == "topics":
        from src.dashboard.pages.topics import render_topics_page
        render_topics_page(sidebar_state)

    elif page_key == "characters":
        from src.dashboard.pages.characters import render_characters_page
        render_characters_page(sidebar_state)

    elif page_key == "comparison":
        from src.dashboard.pages.comparison import render_comparison_page
        render_comparison_page(books, sidebar_state)


if __name__ == "__main__":
    main()
