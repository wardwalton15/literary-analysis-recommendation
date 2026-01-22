"""Sentiment Analysis view page."""

import streamlit as st
from typing import Dict, Any

from src.dashboard.utils.data_loader import get_sentiment_data, get_book_by_id
from src.dashboard.components.charts import (
    create_gauge_chart,
    create_emotional_arc_chart,
    create_pie_chart,
    create_histogram,
)


def render_sentiment_page(sidebar_state: Dict[str, Any]) -> None:
    """Render the sentiment analysis page."""

    book_id = sidebar_state.get("book_id")
    book_title = sidebar_state.get("book_title", "Selected Work")

    st.header(f"💭 Sentiment Analysis: {book_title}")

    if not book_id:
        st.warning("Please select a book from the sidebar.")
        return

    # Load sentiment data
    sentiment_data = get_sentiment_data(book_id)

    if not sentiment_data:
        st.error("No sentiment analysis data available for this book.")
        return

    # Section 1: Overall Sentiment Summary
    st.subheader("📊 Overall Sentiment Summary")

    overall = sentiment_data.get("overall", {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        polarity = overall.get("polarity", 0)
        fig = create_gauge_chart(
            value=polarity,
            title="Polarity",
            min_val=-1,
            max_val=1,
            color_scale="sentiment"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        subjectivity = overall.get("subjectivity", 0)
        fig = create_gauge_chart(
            value=subjectivity,
            title="Subjectivity",
            min_val=0,
            max_val=1,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        compound = overall.get("vader_compound", 0)
        fig = create_gauge_chart(
            value=compound,
            title="VADER Compound",
            min_val=-1,
            max_val=1,
            color_scale="sentiment"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Sentiment classification badge
        if polarity > 0.05:
            classification = "Positive"
            color = "green"
        elif polarity < -0.05:
            classification = "Negative"
            color = "red"
        else:
            classification = "Neutral"
            color = "gray"

        st.markdown(f"""
        <div style="text-align: center; padding: 20px;">
            <h4>Classification</h4>
            <span style="
                background-color: {color};
                color: white;
                padding: 10px 20px;
                border-radius: 20px;
                font-size: 1.2rem;
                font-weight: bold;
            ">{classification}</span>
        </div>
        """, unsafe_allow_html=True)

    # VADER component scores
    st.markdown("**VADER Component Scores:**")
    vader_cols = st.columns(3)
    with vader_cols[0]:
        st.metric("Positive", f"{overall.get('vader_positive', 0):.3f}")
    with vader_cols[1]:
        st.metric("Negative", f"{overall.get('vader_negative', 0):.3f}")
    with vader_cols[2]:
        st.metric("Neutral", f"{overall.get('vader_neutral', 0):.3f}")

    st.markdown("---")

    # Section 2: Emotional Arc
    st.subheader("📈 Emotional Arc")

    arc_data = sentiment_data.get("emotional_arc", {})
    progression = arc_data.get("progression", [])

    if progression:
        # Convert progression list to format expected by chart
        arc_points = [
            {"position": i * 100 / len(progression), "sentiment": val}
            for i, val in enumerate(progression)
        ]
        fig = create_emotional_arc_chart(arc_points, title="Sentiment Through Narrative Progression")
        st.plotly_chart(fig, use_container_width=True)

        # Peaks and valleys annotation
        peaks = arc_data.get("peaks", [])
        valleys = arc_data.get("valleys", [])

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📈 Emotional Peaks**")
            if peaks:
                for i, peak in enumerate(peaks[:3]):
                    position = peak.get("position_ratio", peak.get("index", 0) / len(progression)) * 100
                    value = peak.get("value", 0)
                    st.success(f"""
                    **Peak {i+1}**
                    - Position: {position:.1f}% through narrative
                    - Sentiment Score: {value:.3f}
                    """)
            else:
                st.info("No significant positive peaks detected.")

        with col2:
            st.markdown("**📉 Emotional Valleys**")
            if valleys:
                for i, valley in enumerate(valleys[:3]):
                    position = valley.get("position_ratio", valley.get("index", 0) / len(progression)) * 100
                    value = valley.get("value", 0)
                    st.error(f"""
                    **Valley {i+1}**
                    - Position: {position:.1f}% through narrative
                    - Sentiment Score: {value:.3f}
                    """)
            else:
                st.info("No significant negative valleys detected.")

        # Summary statistics
        avg_sentiment = sum(progression) / len(progression) if progression else 0
        max_sentiment = max(progression) if progression else 0
        min_sentiment = min(progression) if progression else 0

        st.markdown("**Arc Summary:**")
        summary_cols = st.columns(3)
        with summary_cols[0]:
            st.metric("Average Sentiment", f"{avg_sentiment:.3f}")
        with summary_cols[1]:
            st.metric("Maximum", f"{max_sentiment:.3f}")
        with summary_cols[2]:
            st.metric("Minimum", f"{min_sentiment:.3f}")

    else:
        st.info("No emotional arc data available.")

    st.markdown("---")

    # Section 3: Sentence-Level Distribution
    st.subheader("📊 Sentence-Level Distribution")

    sentence_stats = sentiment_data.get("sentence_statistics", {})

    if sentence_stats:
        col1, col2 = st.columns(2)

        with col1:
            # Pie chart
            labels = ["Positive", "Negative", "Neutral"]
            values = [
                sentence_stats.get("positive", 0),
                sentence_stats.get("negative", 0),
                sentence_stats.get("neutral", 0),
            ]
            colors = ["#2ca02c", "#d62728", "#7f7f7f"]

            fig = create_pie_chart(labels, values, "Sentence Sentiment Distribution", colors)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Statistics
            total = sentence_stats.get("total", sum(values))
            st.markdown("**Sentence Counts:**")
            st.metric("Total Sentences", f"{total:,}")

            positive_ratio = sentence_stats.get("positive_ratio", 0)
            negative_ratio = sentence_stats.get("negative_ratio", 0)
            neutral_ratio = 1 - positive_ratio - negative_ratio if positive_ratio + negative_ratio <= 1 else 0

            st.markdown(f"""
            - **Positive:** {values[0]:,} ({positive_ratio*100:.1f}%)
            - **Negative:** {values[1]:,} ({negative_ratio*100:.1f}%)
            - **Neutral:** {values[2]:,} ({neutral_ratio*100:.1f}%)
            """)

        # Histogram of sentiment progression
        if progression:
            fig = create_histogram(
                progression,
                "Distribution of Section Sentiment Scores",
                x_label="Sentiment Score",
                bins=20
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No sentence statistics available.")

    st.markdown("---")

    # Section 4: Interpretation
    st.subheader("📝 Interpretation")

    # Generate interpretation based on data
    interpretation = []

    if polarity < -0.1:
        interpretation.append("The text has an overall **negative tone**, consistent with Kafka's existentialist themes.")
    elif polarity > 0.1:
        interpretation.append("The text shows an overall **positive tone**, which is unusual for Kafka's typically dark narratives.")
    else:
        interpretation.append("The text maintains a **neutral tone** overall, balancing positive and negative elements.")

    if subjectivity > 0.5:
        interpretation.append("The high subjectivity suggests **personal, emotional content** with strong opinions and feelings.")
    else:
        interpretation.append("The low subjectivity indicates more **objective, factual narration**.")

    if progression:
        variance = sum((v - avg_sentiment)**2 for v in progression) / len(progression)
        if variance > 0.1:
            interpretation.append("The **emotional arc shows significant variation**, indicating dramatic shifts in tone throughout the narrative.")
        else:
            interpretation.append("The **emotional arc is relatively stable**, maintaining consistent emotional tone throughout.")

    for interp in interpretation:
        st.markdown(f"- {interp}")
