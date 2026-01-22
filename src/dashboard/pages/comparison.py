"""Comparative Analysis view page."""

import streamlit as st
from typing import Dict, Any, List

from src.dashboard.utils.data_loader import (
    get_comparative_data,
    get_sentiment_data,
    get_complexity_data,
    get_topics_data,
    get_entities_data,
)
from src.dashboard.components.charts import (
    create_grouped_bar_chart,
    create_dual_line_chart,
    create_radar_chart,
    create_heatmap,
)
from src.dashboard.components.network import create_simple_network_stats


def render_comparison_page(books: List[Dict[str, Any]], sidebar_state: Dict[str, Any]) -> None:
    """Render the comparative analysis page."""

    st.header("⚖️ Comparative Analysis")

    if len(books) < 2:
        st.warning("Need at least 2 books for comparison.")
        return

    # Get book selections
    book_titles = [b["title"] for b in books]
    book_ids = {b["title"]: b["id"] for b in books}

    st.markdown("Compare analysis metrics across Kafka's works:")

    col1, col2 = st.columns(2)

    with col1:
        book1_title = st.selectbox("First Work", book_titles, index=0, key="compare_book1")

    with col2:
        remaining_titles = [t for t in book_titles if t != book1_title]
        book2_title = st.selectbox("Second Work", remaining_titles, index=0, key="compare_book2")

    book1_id = book_ids[book1_title]
    book2_id = book_ids[book2_title]

    st.markdown("---")

    # Section 1: Side-by-Side Summary
    st.subheader("📊 Side-by-Side Summary")

    book1 = next((b for b in books if b["id"] == book1_id), {})
    book2 = next((b for b in books if b["id"] == book2_id), {})

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### {book1_title}")

        st.metric("Word Count", f"{book1.get('word_count', 0):,}")

        sentiment1 = get_sentiment_data(book1_id)
        if sentiment1:
            polarity = sentiment1.get("overall", {}).get("polarity", 0)
            st.metric("Overall Sentiment", f"{polarity:.3f}")

        complexity1 = get_complexity_data(book1_id)
        if complexity1:
            flesch = complexity1.get("readability_scores", {}).get("flesch_reading_ease", 0)
            st.metric("Flesch Reading Ease", f"{flesch:.1f}")

        topics1 = get_topics_data(book1_id)
        if topics1:
            theme_labels = [t.get("theme_label", f"Topic {i+1}") for i, t in enumerate(topics1.get("topics", [])[:3])]
            st.markdown(f"**Main Themes:** {', '.join(theme_labels)}")

    with col2:
        st.markdown(f"### {book2_title}")

        st.metric("Word Count", f"{book2.get('word_count', 0):,}")

        sentiment2 = get_sentiment_data(book2_id)
        if sentiment2:
            polarity = sentiment2.get("overall", {}).get("polarity", 0)
            st.metric("Overall Sentiment", f"{polarity:.3f}")

        complexity2 = get_complexity_data(book2_id)
        if complexity2:
            flesch = complexity2.get("readability_scores", {}).get("flesch_reading_ease", 0)
            st.metric("Flesch Reading Ease", f"{flesch:.1f}")

        topics2 = get_topics_data(book2_id)
        if topics2:
            theme_labels = [t.get("theme_label", f"Topic {i+1}") for i, t in enumerate(topics2.get("topics", [])[:3])]
            st.markdown(f"**Main Themes:** {', '.join(theme_labels)}")

    st.markdown("---")

    # Section 2: Sentiment Comparison
    st.subheader("💭 Sentiment Comparison")

    if not sentiment1 or not sentiment2:
        st.info("Sentiment data not available for one or both works.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            # Emotional Arc Overlay
            arc_data1 = sentiment1.get("emotional_arc", {})
            arc_data2 = sentiment2.get("emotional_arc", {})
            progression1 = arc_data1.get("progression", []) if isinstance(arc_data1, dict) else []
            progression2 = arc_data2.get("progression", []) if isinstance(arc_data2, dict) else []

            if progression1 and progression2:
                # Convert to format expected by chart (position, sentiment)
                arc1 = [{"position": i * 100 / len(progression1), "sentiment": val} for i, val in enumerate(progression1)]
                arc2 = [{"position": i * 100 / len(progression2), "sentiment": val} for i, val in enumerate(progression2)]

                fig = create_dual_line_chart(
                    arc1, arc2,
                    book1_title, book2_title,
                    "Emotional Arc Comparison"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Emotional arc data not available for comparison.")

        with col2:
            # Bar chart of sentiment components
            categories = ["Polarity", "Subjectivity", "VADER Compound"]
            overall1 = sentiment1.get("overall", {})
            overall2 = sentiment2.get("overall", {})
            values1 = [
                overall1.get("polarity", 0),
                overall1.get("subjectivity", 0),
                overall1.get("vader_compound", 0),
            ]
            values2 = [
                overall2.get("polarity", 0),
                overall2.get("subjectivity", 0),
                overall2.get("vader_compound", 0),
            ]

            fig = create_grouped_bar_chart(
                categories,
                {book1_title: values1, book2_title: values2},
                "Sentiment Metrics Comparison",
                "Score"
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Section 3: Complexity Comparison
    st.subheader("📊 Complexity Comparison")

    if complexity1 and complexity2:
        # Grouped bar chart of all readability scores
        readability_metrics = [
            "flesch_reading_ease",
            "flesch_kincaid_grade",
            "gunning_fog",
            "smog_index",
            "coleman_liau_index",
            "automated_readability_index",
        ]

        metric_labels = [
            "Flesch Ease",
            "FK Grade",
            "Gunning Fog",
            "SMOG",
            "Coleman-Liau",
            "ARI",
        ]

        r1 = complexity1.get("readability_scores", {})
        r2 = complexity2.get("readability_scores", {})

        values1 = [r1.get(m, 0) for m in readability_metrics]
        values2 = [r2.get(m, 0) for m in readability_metrics]

        col1, col2 = st.columns(2)

        with col1:
            fig = create_grouped_bar_chart(
                metric_labels,
                {book1_title: values1, book2_title: values2},
                "Readability Scores Comparison",
                "Score"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Radar chart comparison
            # Normalize values for radar chart
            max_values = [100, 20, 20, 20, 20, 20]
            norm1 = [v / m for v, m in zip(values1, max_values)]
            norm2 = [v / m for v, m in zip(values2, max_values)]

            import plotly.graph_objects as go

            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=norm1 + [norm1[0]],
                theta=metric_labels + [metric_labels[0]],
                fill='toself',
                name=book1_title,
                fillcolor='rgba(31, 119, 180, 0.3)',
                line=dict(color='rgb(31, 119, 180)')
            ))

            fig.add_trace(go.Scatterpolar(
                r=norm2 + [norm2[0]],
                theta=metric_labels + [metric_labels[0]],
                fill='toself',
                name=book2_title,
                fillcolor='rgba(255, 127, 14, 0.3)',
                line=dict(color='rgb(255, 127, 14)')
            ))

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="Normalized Complexity Radar",
                height=400,
            )

            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Section 4: Theme Comparison
    st.subheader("🎯 Theme Comparison")

    if topics1 and topics2:
        # Create heatmap of topic presence
        topics_list1 = topics1.get("topics", [])
        topics_list2 = topics2.get("topics", [])

        # Get all unique theme labels
        all_themes = list(set(
            [t.get("theme_label", f"Topic {i}") for i, t in enumerate(topics_list1)] +
            [t.get("theme_label", f"Topic {i}") for i, t in enumerate(topics_list2)]
        ))

        # Build proportion matrix
        def get_theme_proportion(topics_list, theme):
            for t in topics_list:
                if t.get("theme_label", "") == theme:
                    return t.get("weight", t.get("document_proportion", 0))
            return 0

        values_matrix = [
            [
                get_theme_proportion(topics_list1, theme),
                get_theme_proportion(topics_list2, theme)
            ]
            for theme in all_themes
        ]

        fig = create_heatmap(
            x_labels=[book1_title, book2_title],
            y_labels=all_themes,
            values=values_matrix,
            title="Topic Distribution Heatmap",
            color_scale="Greens"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Shared vs Unique themes
        col1, col2, col3 = st.columns(3)

        themes1 = set(t.get("theme_label", "") for t in topics_list1)
        themes2 = set(t.get("theme_label", "") for t in topics_list2)

        shared = themes1 & themes2
        unique1 = themes1 - themes2
        unique2 = themes2 - themes1

        with col1:
            st.markdown(f"**Shared Themes ({len(shared)}):**")
            for t in shared:
                st.markdown(f"- {t}")

        with col2:
            st.markdown(f"**Unique to {book1_title} ({len(unique1)}):**")
            for t in unique1:
                st.markdown(f"- {t}")

        with col3:
            st.markdown(f"**Unique to {book2_title} ({len(unique2)}):**")
            for t in unique2:
                st.markdown(f"- {t}")

    st.markdown("---")

    # Section 5: Character Network Comparison
    st.subheader("🔗 Character Network Comparison")

    entities1 = get_entities_data(book1_id)
    entities2 = get_entities_data(book2_id)

    if entities1 and entities2:
        stats1 = create_simple_network_stats(entities1)
        stats2 = create_simple_network_stats(entities2)

        metrics = ["total_characters", "total_relationships", "network_density", "num_communities"]
        labels = ["Characters", "Relationships", "Density", "Communities"]

        col1, col2, col3, col4 = st.columns(4)

        columns = [col1, col2, col3, col4]

        for i, (metric, label) in enumerate(zip(metrics, labels)):
            with columns[i]:
                val1 = stats1.get(metric, 0)
                val2 = stats2.get(metric, 0)

                if isinstance(val1, float):
                    st.metric(f"{label} ({book1_title[:15]}...)", f"{val1:.3f}")
                    st.metric(f"{label} ({book2_title[:15]}...)", f"{val2:.3f}")
                else:
                    st.metric(f"{label} ({book1_title[:15]}...)", val1)
                    st.metric(f"{label} ({book2_title[:15]}...)", val2)

        # Most central characters comparison
        st.markdown("**Most Central Characters:**")
        col1, col2 = st.columns(2)

        with col1:
            chars1 = entities1.get("central_characters", [])
            top_chars1 = sorted(chars1, key=lambda x: x.get("degree_centrality", 0), reverse=True)[:5]
            st.markdown(f"**{book1_title}:**")
            for c in top_chars1:
                st.markdown(f"- {c.get('name', c.get('entity', ''))} ({c.get('degree_centrality', 0):.3f})")

        with col2:
            chars2 = entities2.get("central_characters", [])
            top_chars2 = sorted(chars2, key=lambda x: x.get("degree_centrality", 0), reverse=True)[:5]
            st.markdown(f"**{book2_title}:**")
            for c in top_chars2:
                st.markdown(f"- {c.get('name', c.get('entity', ''))} ({c.get('degree_centrality', 0):.3f})")
