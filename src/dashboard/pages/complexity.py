"""Complexity Metrics view page."""

import streamlit as st
from typing import Dict, Any
import pandas as pd

from src.dashboard.utils.data_loader import get_complexity_data
from src.dashboard.components.charts import (
    create_gauge_chart,
    create_bar_chart,
    create_radar_chart,
)


def get_reading_level(score: float) -> str:
    """Interpret Flesch Reading Ease score."""
    if score >= 90:
        return "Very Easy (5th Grade)"
    elif score >= 80:
        return "Easy (6th Grade)"
    elif score >= 70:
        return "Fairly Easy (7th Grade)"
    elif score >= 60:
        return "Standard (8th-9th Grade)"
    elif score >= 50:
        return "Fairly Difficult (10th-12th Grade)"
    elif score >= 30:
        return "Difficult (College)"
    else:
        return "Very Difficult (College Graduate)"


def render_complexity_page(sidebar_state: Dict[str, Any]) -> None:
    """Render the complexity metrics page."""

    book_id = sidebar_state.get("book_id")
    book_title = sidebar_state.get("book_title", "Selected Work")

    st.header(f"📊 Complexity Metrics: {book_title}")

    if not book_id:
        st.warning("Please select a book from the sidebar.")
        return

    # Load complexity data
    complexity_data = get_complexity_data(book_id)

    if not complexity_data:
        st.error("No complexity analysis data available for this book.")
        return

    # Section 1: Readability Scores Dashboard
    st.subheader("📖 Readability Scores")

    readability = complexity_data.get("readability", {})

    # Main metrics in gauges
    col1, col2, col3 = st.columns(3)

    with col1:
        flesch = readability.get("flesch_reading_ease", 0)
        fig = create_gauge_chart(
            value=flesch,
            title="Flesch Reading Ease",
            min_val=0,
            max_val=100,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"**{get_reading_level(flesch)}**")

    with col2:
        fk_grade = readability.get("flesch_kincaid_grade", 0)
        fig = create_gauge_chart(
            value=fk_grade,
            title="Flesch-Kincaid Grade",
            min_val=0,
            max_val=20,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"**Grade Level: {fk_grade:.1f}**")

    with col3:
        gunning_fog = readability.get("gunning_fog", 0)
        fig = create_gauge_chart(
            value=gunning_fog,
            title="Gunning Fog Index",
            min_val=0,
            max_val=20,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"**Years of Education: {gunning_fog:.1f}**")

    # Additional readability scores
    st.markdown("**Additional Readability Metrics:**")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        smog = readability.get("smog_index", 0)
        st.metric("SMOG Index", f"{smog:.1f}", help="Simple Measure of Gobbledygook")

    with col2:
        coleman = readability.get("coleman_liau_index", 0)
        st.metric("Coleman-Liau", f"{coleman:.1f}", help="Based on character counts")

    with col3:
        ari = readability.get("automated_readability_index", 0)
        st.metric("ARI", f"{ari:.1f}", help="Automated Readability Index")

    with col4:
        dale_chall = readability.get("dale_chall_score", 0)
        st.metric("Dale-Chall", f"{dale_chall:.2f}", help="Uses familiar word list")

    st.markdown("---")

    # Section 2: Text Statistics
    st.subheader("📝 Text Statistics")

    text_stats = complexity_data.get("text_statistics", {})

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Word Count",
            f"{text_stats.get('word_count', 0):,}",
            help="Total number of words"
        )

    with col2:
        st.metric(
            "Sentence Count",
            f"{text_stats.get('sentence_count', 0):,}",
            help="Total number of sentences"
        )

    with col3:
        avg_sentence = text_stats.get('avg_sentence_length', 0)
        st.metric(
            "Avg Sentence Length",
            f"{avg_sentence:.1f} words",
            help="Average words per sentence"
        )

    with col4:
        avg_syllables = text_stats.get('avg_syllables_per_word', 0)
        st.metric(
            "Avg Syllables/Word",
            f"{avg_syllables:.2f}",
            help="Average syllables per word"
        )

    with col5:
        avg_word_len = text_stats.get('avg_word_length', 0)
        st.metric(
            "Avg Word Length",
            f"{avg_word_len:.2f} chars",
            help="Average characters per word"
        )

    # Additional statistics
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Text Counts:**")
        char_count = text_stats.get('character_count', 0)
        syllable_count = text_stats.get('syllable_count', 0)
        st.markdown(f"""
        - **Characters:** {char_count:,}
        - **Syllables:** {syllable_count:,}
        """)

    with col2:
        st.markdown("**Averages:**")
        st.markdown(f"""
        - **Words per Sentence:** {avg_sentence:.1f}
        - **Syllables per Word:** {avg_syllables:.2f}
        - **Characters per Word:** {avg_word_len:.2f}
        """)

    st.markdown("---")

    # Section 3: Vocabulary Analysis
    st.subheader("🔤 Vocabulary Analysis")

    vocab_stats = complexity_data.get("vocabulary", {})

    col1, col2 = st.columns(2)

    with col1:
        # Type-Token Ratio and related metrics
        ttr = vocab_stats.get("type_token_ratio", 0)
        fig = create_gauge_chart(
            value=ttr * 100,
            title="Type-Token Ratio",
            min_val=0,
            max_val=100,
            suffix="%"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Higher = more diverse vocabulary")

        st.markdown("**Vocabulary Metrics:**")
        unique_words = vocab_stats.get("unique_words", 0)
        hapax = vocab_stats.get("hapax_legomena", 0)
        hapax_ratio = vocab_stats.get("hapax_ratio", 0)
        st.markdown(f"""
        - **Unique Words:** {unique_words:,}
        - **Hapax Legomena:** {hapax:,}
          *(words appearing only once)*
        - **Hapax Ratio:** {hapax_ratio:.2%}
        """)

    with col2:
        # Vocabulary richness metrics
        richness = vocab_stats.get("vocabulary_richness", 0)
        st.metric(
            "Vocabulary Richness",
            f"{richness:.4f}",
            help="Measure of vocabulary diversity"
        )

        # Word length distribution
        word_length_dist = complexity_data.get("word_length_distribution", {})
        if word_length_dist:
            lengths = list(word_length_dist.keys())
            counts = list(word_length_dist.values())

            # Sort by length
            sorted_items = sorted(zip(lengths, counts), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0)
            lengths = [str(l) for l, _ in sorted_items[:15]]
            counts = [c for _, c in sorted_items[:15]]

            fig = create_bar_chart(
                x=lengths,
                y=counts,
                title="Word Length Distribution",
                x_label="Word Length (characters)",
                y_label="Count"
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Section 4: Complexity Through Narrative
    st.subheader("📈 Complexity Through Narrative")

    section_complexity = complexity_data.get("section_complexity", [])

    if section_complexity:
        # Create line chart showing complexity progression
        import plotly.graph_objects as go

        positions = [s.get("position", i*10) for i, s in enumerate(section_complexity)]
        flesch_scores = [s.get("flesch_reading_ease", 0) for s in section_complexity]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=positions,
            y=flesch_scores,
            mode='lines+markers',
            name='Flesch Reading Ease',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=8),
            hovertemplate="Position: %{x}%<br>Score: %{y:.1f}<extra></extra>"
        ))

        fig.update_layout(
            title="Readability Score Through Narrative",
            xaxis_title="Narrative Progress (%)",
            yaxis_title="Flesch Reading Ease",
            height=400,
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Identify most/least complex passages
        col1, col2 = st.columns(2)

        with col1:
            # Most complex (lowest Flesch score)
            most_complex = min(section_complexity, key=lambda x: x.get("flesch_reading_ease", 100))
            st.markdown("**📕 Most Complex Passage**")
            st.warning(f"""
            - Position: {most_complex.get('position', 'N/A')}%
            - Flesch Score: {most_complex.get('flesch_reading_ease', 0):.1f}
            - Reading Level: {get_reading_level(most_complex.get('flesch_reading_ease', 0))}
            """)

        with col2:
            # Least complex (highest Flesch score)
            least_complex = max(section_complexity, key=lambda x: x.get("flesch_reading_ease", 0))
            st.markdown("**📗 Easiest Passage**")
            st.success(f"""
            - Position: {least_complex.get('position', 'N/A')}%
            - Flesch Score: {least_complex.get('flesch_reading_ease', 0):.1f}
            - Reading Level: {get_reading_level(least_complex.get('flesch_reading_ease', 0))}
            """)

    else:
        st.info("Section-by-section complexity data not available.")

    # Summary radar chart
    st.markdown("---")
    st.subheader("🎯 Complexity Profile")

    # Normalize metrics for radar chart
    metrics = {
        "Flesch Ease": min(readability.get("flesch_reading_ease", 0) / 100, 1),
        "Grade Level": min(readability.get("flesch_kincaid_grade", 0) / 20, 1),
        "Vocab Richness": min(ttr, 1),
        "Sentence Length": min(text_stats.get("average_sentence_length", 0) / 30, 1),
        "Word Complexity": min(avg_syllables / 3, 1),
    }

    fig = create_radar_chart(
        categories=list(metrics.keys()),
        values=list(metrics.values()),
        title="Normalized Complexity Profile"
    )
    st.plotly_chart(fig, use_container_width=True)
