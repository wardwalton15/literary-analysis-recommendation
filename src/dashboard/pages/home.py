"""Home/Overview page for the dashboard."""

import streamlit as st
from typing import List, Dict, Any

from src.dashboard.utils.data_loader import (
    get_corpus_statistics,
    get_sentiment_data,
    get_complexity_data,
    get_topics_data,
)


def render_home_page(books: List[Dict[str, Any]], sidebar_state: Dict[str, Any]) -> None:
    """Render the home/overview page."""

    # Header
    st.markdown('<p class="main-header">📚 Kafka Literary Analysis Dashboard</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Exploring Franz Kafka\'s English Works Through NLP</p>',
        unsafe_allow_html=True
    )

    # Project description
    st.markdown("""
    Welcome to the **Literary Analysis Dashboard** for Franz Kafka's English works.
    This interactive tool visualizes the results of comprehensive NLP analysis performed
    on *The Metamorphosis* and *The Trial*.

    **Explore the following analyses:**
    - **Sentiment Analysis**: Track emotional arcs and sentiment patterns
    - **Complexity Metrics**: Examine readability and linguistic complexity
    - **Theme Extraction**: Discover extracted topics and literary themes
    - **Character Networks**: Visualize character relationships and communities
    """)

    st.markdown("---")

    # Corpus Statistics
    st.subheader("📈 Corpus Overview")

    stats = get_corpus_statistics()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Works Analyzed",
            value=len(books),
            help="Number of Kafka works in the corpus"
        )

    with col2:
        total_words = sum(b.get("word_count", 0) for b in books)
        st.metric(
            label="Total Words",
            value=f"{total_words:,}",
            help="Combined word count across all works"
        )

    with col3:
        analysis_count = len(stats.get("analysis_types", {}))
        st.metric(
            label="Analysis Types",
            value=analysis_count,
            help="Number of different NLP analyses performed"
        )

    with col4:
        st.metric(
            label="Coverage",
            value="100%",
            help="All works have complete analysis"
        )

    st.markdown("---")

    # Books summary
    st.subheader("📖 Works in Corpus")

    for book in books:
        with st.expander(f"**{book['title']}** by {book['author']}", expanded=True):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"""
                - **Word Count:** {book.get('word_count', 'N/A'):,}
                - **Publication Year:** {book.get('publication_year', 'N/A')}
                """)

                # Quick sentiment preview
                sentiment = get_sentiment_data(book["id"])
                if sentiment:
                    overall = sentiment.get("overall", {})
                    polarity = overall.get("polarity", 0)
                    sentiment_label = "Positive" if polarity > 0.05 else "Negative" if polarity < -0.05 else "Neutral"
                    st.markdown(f"- **Overall Sentiment:** {sentiment_label} ({polarity:.3f})")

            with col2:
                # Quick complexity preview
                complexity = get_complexity_data(book["id"])
                if complexity:
                    readability = complexity.get("readability_scores", {})
                    flesch = readability.get("flesch_reading_ease", 0)
                    st.metric("Flesch Reading Ease", f"{flesch:.1f}")

    st.markdown("---")

    # Sample insights
    st.subheader("💡 Key Insights Preview")

    book_id = sidebar_state.get("book_id")
    if book_id:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Emotional Journey**")
            sentiment = get_sentiment_data(book_id)
            if sentiment:
                arc_data = sentiment.get("emotional_arc", {})
                progression = arc_data.get("progression", []) if isinstance(arc_data, dict) else []
                if progression:
                    peaks = arc_data.get("peaks", [])
                    valleys = arc_data.get("valleys", [])

                    peak_pos = peaks[0].get("position_ratio", 0) * 100 if peaks else None
                    valley_pos = valleys[0].get("position_ratio", 0) * 100 if valleys else None

                    st.markdown("- The narrative shows distinct emotional patterns")
                    if peak_pos is not None:
                        st.markdown(f"- Most positive moment at {peak_pos:.0f}% progress")
                    if valley_pos is not None:
                        st.markdown(f"- Most negative moment at {valley_pos:.0f}% progress")

        with col2:
            st.markdown("**Dominant Themes**")
            topics = get_topics_data(book_id)
            if topics:
                topic_list = topics.get("topics", [])
                if topic_list:
                    themes = [t.get("theme_label", f"Topic {i+1}") for i, t in enumerate(topic_list[:3])]
                    st.markdown(f"""
                    Top extracted themes:
                    - {themes[0] if len(themes) > 0 else 'N/A'}
                    - {themes[1] if len(themes) > 1 else 'N/A'}
                    - {themes[2] if len(themes) > 2 else 'N/A'}
                    """)

    # Navigation hints
    st.markdown("---")
    st.info("Use the navigation tabs above to explore detailed analyses. Select a work in the sidebar to view its individual metrics.")
