"""Themes & Topics view page."""

import streamlit as st
from typing import Dict, Any, List
import pandas as pd

from src.dashboard.utils.data_loader import get_topics_data
from src.dashboard.components.charts import (
    create_bar_chart,
    create_radar_chart,
    create_word_cloud_placeholder,
)


# Kafka's canonical literary themes for comparison
KAFKA_THEMES = [
    "Alienation",
    "Bureaucracy",
    "Transformation",
    "Guilt",
    "Family",
    "Anxiety/Absurdity"
]


def render_topics_page(sidebar_state: Dict[str, Any]) -> None:
    """Render the themes and topics page."""

    book_id = sidebar_state.get("book_id")
    book_title = sidebar_state.get("book_title", "Selected Work")

    st.header(f"🎯 Themes & Topics: {book_title}")

    if not book_id:
        st.warning("Please select a book from the sidebar.")
        return

    # Load topics data
    topics_data = get_topics_data(book_id)

    if not topics_data:
        st.error("No topic modeling data available for this book.")
        return

    # Section 1: Topic Overview
    st.subheader("📋 Topic Modeling Overview")

    topics_list = topics_data.get("topics", [])
    theme_labels = topics_data.get("theme_labels", [])
    num_topics = topics_data.get("num_topics", len(topics_list))

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Topics Extracted",
            num_topics,
            help="Number of topics identified"
        )

    with col2:
        model_type = topics_data.get("model_type", "LDA")
        st.metric(
            "Model Type",
            model_type,
            help="Topic modeling algorithm used"
        )

    with col3:
        coherence = topics_data.get("coherence_score", 0)
        st.metric(
            "Coherence Score",
            f"{coherence:.3f}" if coherence else "N/A",
            help="Model quality metric (higher is better)"
        )

    with col4:
        # Dominant topic distribution
        dist = topics_data.get("dominant_topic_distribution", {})
        total_docs = sum(dist.values()) if dist else 0
        st.metric(
            "Documents Analyzed",
            total_docs,
            help="Number of text chunks analyzed"
        )

    st.markdown("---")

    # Section 2: Topic Cards
    st.subheader("📚 Extracted Topics")

    if not topics_list:
        st.info("No topics extracted.")
        return

    # Get dominant topic distribution for calculating proportions
    dist = topics_data.get("dominant_topic_distribution", {})
    total_docs = sum(dist.values()) if dist else 1

    # Display topics in expandable cards
    for i, topic in enumerate(topics_list):
        # Get theme label from separate list or generate one
        if i < len(theme_labels):
            theme_label = theme_labels[i]
        else:
            theme_label = f"Topic {i+1}"

        topic_words = topic.get("words", [])

        # Calculate proportion from dominant topic distribution
        topic_key = str(topic.get("id", i))
        topic_count = dist.get(topic_key, 0)
        weight = topic_count / total_docs if total_docs > 0 else 0

        with st.expander(f"**{theme_label}** ({weight*100:.1f}% dominant)", expanded=(i < 2)):
            col1, col2 = st.columns([2, 1])

            with col1:
                # Word cloud for this topic
                if topic_words:
                    word_list = []
                    for w in topic_words[:15]:
                        if isinstance(w, list) and len(w) >= 2:
                            word_list.append({"word": w[0], "weight": w[1]})
                        elif isinstance(w, dict):
                            word_list.append({"word": w.get("word", ""), "weight": w.get("weight", 0.1)})
                        elif isinstance(w, tuple):
                            word_list.append({"word": w[0], "weight": w[1]})
                        else:
                            word_list.append({"word": str(w), "weight": 0.1})

                    fig = create_word_cloud_placeholder(word_list, f"Key Words: {theme_label}")
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("**Top Words:**")
                # Create table of words
                word_data = []
                for j, w in enumerate(topic_words[:10]):
                    if isinstance(w, list) and len(w) >= 2:
                        word_data.append({
                            "Rank": j+1,
                            "Word": w[0],
                            "Weight": f"{w[1]:.4f}"
                        })
                    elif isinstance(w, dict):
                        word_data.append({
                            "Rank": j+1,
                            "Word": w.get("word", ""),
                            "Weight": f"{w.get('weight', 0):.4f}"
                        })
                    elif isinstance(w, tuple):
                        word_data.append({
                            "Rank": j+1,
                            "Word": w[0],
                            "Weight": f"{w[1]:.4f}"
                        })
                    else:
                        word_data.append({
                            "Rank": j+1,
                            "Word": str(w),
                            "Weight": "N/A"
                        })

                if word_data:
                    st.dataframe(pd.DataFrame(word_data), hide_index=True)

    st.markdown("---")

    # Section 3: Topic Distribution
    st.subheader("📊 Topic Distribution")

    # Overall topic distribution bar chart
    dist_labels = []
    proportions = []

    for i, topic in enumerate(topics_list):
        if i < len(theme_labels):
            label = theme_labels[i]
        else:
            label = f"Topic {i+1}"
        dist_labels.append(label)

        topic_key = str(topic.get("id", i))
        topic_count = dist.get(topic_key, 0)
        proportions.append(topic_count / total_docs * 100 if total_docs > 0 else 0)

    if dist_labels and proportions:
        fig = create_bar_chart(
            x=dist_labels,
            y=proportions,
            title="Topic Distribution Across Text",
            x_label="Topic",
            y_label="Percentage (%)",
            color="#2ca02c"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Topic distribution through narrative (if available)
    chunk_distribution = topics_data.get("chunk_distribution", [])
    if chunk_distribution:
        st.markdown("**Topic Prevalence Through Narrative:**")

        import plotly.graph_objects as go

        fig = go.Figure()

        # Add a trace for each topic
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for i, topic in enumerate(topics_list[:5]):
            topic_label = topic.get("theme_label", f"Topic {i+1}")
            values = [chunk.get(f"topic_{i}", 0) for chunk in chunk_distribution]
            positions = [chunk.get("position", j*10) for j, chunk in enumerate(chunk_distribution)]

            fig.add_trace(go.Scatter(
                x=positions,
                y=values,
                mode='lines',
                name=topic_label,
                line=dict(color=colors[i % len(colors)], width=2),
                stackgroup='one'  # Stacked area
            ))

        fig.update_layout(
            title="Topic Distribution Through Narrative",
            xaxis_title="Narrative Progress (%)",
            yaxis_title="Topic Proportion",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Section 4: Literary Theme Mapping
    st.subheader("📖 Literary Theme Mapping")

    st.markdown("""
    Comparing extracted topics with Kafka's canonical literary themes:
    """)

    # Theme alignment scores (from analysis or computed)
    theme_alignment = topics_data.get("theme_alignment", {})

    if theme_alignment:
        scores = [theme_alignment.get(theme, 0) for theme in KAFKA_THEMES]
    else:
        # Compute basic alignment based on topic labels matching canonical themes
        scores = []
        for theme in KAFKA_THEMES:
            score = 0
            theme_lower = theme.lower()
            for topic in topics_list:
                topic_label = topic.get("theme_label", "").lower()
                if theme_lower in topic_label or topic_label in theme_lower:
                    score = max(score, topic.get("weight", topic.get("document_proportion", 0)))
                # Also check top words
                for w in topic.get("words", topic.get("top_words", []))[:10]:
                    word = w.get("word", w) if isinstance(w, dict) else (w[0] if isinstance(w, tuple) else str(w))
                    if theme_lower in word.lower():
                        score = max(score, 0.3)
            scores.append(score)

    col1, col2 = st.columns([2, 1])

    with col1:
        # Radar chart
        fig = create_radar_chart(
            categories=KAFKA_THEMES,
            values=scores,
            title="Alignment with Kafka's Canonical Themes",
            fill_color="rgba(44, 160, 44, 0.5)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Theme Scores:**")
        for theme, score in zip(KAFKA_THEMES, scores):
            # Color-coded progress bars
            st.progress(min(score, 1.0), text=f"{theme}: {score*100:.1f}%")

    # Interpretation
    st.markdown("---")
    st.markdown("**📝 Interpretation:**")

    dominant_themes = sorted(zip(KAFKA_THEMES, scores), key=lambda x: x[1], reverse=True)[:3]
    weak_themes = sorted(zip(KAFKA_THEMES, scores), key=lambda x: x[1])[:2]

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"""
        **Strongest Themes:**
        - {dominant_themes[0][0]} ({dominant_themes[0][1]*100:.1f}%)
        - {dominant_themes[1][0]} ({dominant_themes[1][1]*100:.1f}%)
        - {dominant_themes[2][0]} ({dominant_themes[2][1]*100:.1f}%)
        """)

    with col2:
        st.info(f"""
        **Less Prominent:**
        - {weak_themes[0][0]} ({weak_themes[0][1]*100:.1f}%)
        - {weak_themes[1][0]} ({weak_themes[1][1]*100:.1f}%)
        """)
