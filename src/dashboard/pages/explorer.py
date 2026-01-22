"""Data Explorer view page."""

import streamlit as st
from typing import Dict, Any
import json
import pandas as pd

from src.dashboard.utils.data_loader import (
    get_sentiment_data,
    get_complexity_data,
    get_topics_data,
    get_entities_data,
    get_preprocessing_data,
    get_book_by_id,
    get_all_analysis_types,
)


def render_explorer_page(sidebar_state: Dict[str, Any]) -> None:
    """Render the data explorer page."""

    book_id = sidebar_state.get("book_id")
    book_title = sidebar_state.get("book_title", "Selected Work")

    st.header(f"📁 Data Explorer: {book_title}")

    if not book_id:
        st.warning("Please select a book from the sidebar.")
        return

    st.markdown("""
    Explore raw analysis data, export to various formats, and query the underlying database.
    """)

    # Tabs for different explorer functions
    tab1, tab2, tab3 = st.tabs(["📄 JSON Viewer", "⬇️ Export Data", "🔍 Analysis Details"])

    # Tab 1: JSON Viewer
    with tab1:
        st.subheader("Raw Analysis Data")

        analysis_types = get_all_analysis_types(book_id)

        if not analysis_types:
            st.info("No analysis data available for this book.")
        else:
            selected_type = st.selectbox(
                "Select Analysis Type",
                analysis_types,
                key="json_viewer_type"
            )

            # Load the selected analysis data
            data_loaders = {
                "sentiment": get_sentiment_data,
                "complexity": get_complexity_data,
                "topics": get_topics_data,
                "entities": get_entities_data,
                "preprocessing": get_preprocessing_data,
            }

            loader = data_loaders.get(selected_type)
            if loader:
                data = loader(book_id)

                if data:
                    # Display formatted JSON
                    st.json(data)

                    # Option to expand/collapse
                    with st.expander("View as formatted text"):
                        st.code(json.dumps(data, indent=2), language="json")
                else:
                    st.warning(f"No {selected_type} data available.")

    # Tab 2: Export Data
    with tab2:
        st.subheader("Export Analysis Data")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 CSV Exports")

            # Export sentiment data as CSV
            sentiment_data = get_sentiment_data(book_id)
            if sentiment_data:
                # Emotional arc as CSV
                arc_data = sentiment_data.get("emotional_arc", [])
                if arc_data:
                    arc_df = pd.DataFrame(arc_data)
                    csv_arc = arc_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Emotional Arc (CSV)",
                        data=csv_arc,
                        file_name=f"{book_title.replace(' ', '_')}_emotional_arc.csv",
                        mime="text/csv"
                    )

            # Export complexity data as CSV
            complexity_data = get_complexity_data(book_id)
            if complexity_data:
                readability = complexity_data.get("readability_scores", {})
                text_stats = complexity_data.get("text_statistics", {})

                combined = {**readability, **text_stats}
                complexity_df = pd.DataFrame([combined])
                csv_complexity = complexity_df.to_csv(index=False)

                st.download_button(
                    label="📥 Download Complexity Metrics (CSV)",
                    data=csv_complexity,
                    file_name=f"{book_title.replace(' ', '_')}_complexity.csv",
                    mime="text/csv"
                )

            # Export character data as CSV
            entities_data = get_entities_data(book_id)
            if entities_data:
                characters = entities_data.get("central_characters", [])
                if characters:
                    char_df = pd.DataFrame(characters)
                    csv_chars = char_df.to_csv(index=False)

                    st.download_button(
                        label="📥 Download Characters (CSV)",
                        data=csv_chars,
                        file_name=f"{book_title.replace(' ', '_')}_characters.csv",
                        mime="text/csv"
                    )

        with col2:
            st.markdown("### 📄 JSON Exports")

            # Full analysis JSON
            all_data = {
                "book_id": book_id,
                "book_title": book_title,
                "sentiment": get_sentiment_data(book_id),
                "complexity": get_complexity_data(book_id),
                "topics": get_topics_data(book_id),
                "entities": get_entities_data(book_id),
                "preprocessing": get_preprocessing_data(book_id),
            }

            json_full = json.dumps(all_data, indent=2)

            st.download_button(
                label="📥 Download Full Analysis (JSON)",
                data=json_full,
                file_name=f"{book_title.replace(' ', '_')}_full_analysis.json",
                mime="application/json"
            )

            # Individual analysis exports
            for analysis_type in ["sentiment", "complexity", "topics", "entities"]:
                data = all_data.get(analysis_type)
                if data:
                    json_data = json.dumps(data, indent=2)
                    st.download_button(
                        label=f"📥 Download {analysis_type.capitalize()} (JSON)",
                        data=json_data,
                        file_name=f"{book_title.replace(' ', '_')}_{analysis_type}.json",
                        mime="application/json",
                        key=f"export_{analysis_type}"
                    )

            # Network export (GEXF format for Gephi)
            if entities_data:
                st.markdown("### 🕸️ Network Export")

                # Create simple GEXF XML
                gexf_content = create_gexf_export(entities_data, book_title)

                st.download_button(
                    label="📥 Download Network (GEXF)",
                    data=gexf_content,
                    file_name=f"{book_title.replace(' ', '_')}_network.gexf",
                    mime="application/xml"
                )

    # Tab 3: Analysis Details
    with tab3:
        st.subheader("Analysis Summary")

        book_info = get_book_by_id(book_id)

        if book_info:
            st.markdown("### 📖 Book Information")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"""
                - **Title:** {book_info.get('title', 'N/A')}
                - **Author:** {book_info.get('author', 'N/A')}
                - **Publication Year:** {book_info.get('publication_year', 'N/A')}
                """)

            with col2:
                st.markdown(f"""
                - **Word Count:** {book_info.get('word_count', 0):,}
                - **Language:** {book_info.get('language', 'N/A')}
                - **Book ID:** {book_id}
                """)

        st.markdown("---")

        st.markdown("### 📊 Available Analyses")

        analysis_types = get_all_analysis_types(book_id)

        for analysis_type in analysis_types:
            with st.expander(f"✅ {analysis_type.capitalize()}", expanded=False):
                data = None
                if analysis_type == "sentiment":
                    data = get_sentiment_data(book_id)
                    if data:
                        st.markdown(f"""
                        - **Overall Polarity:** {data.get('overall', {}).get('polarity', 0):.3f}
                        - **Subjectivity:** {data.get('overall', {}).get('subjectivity', 0):.3f}
                        - **Emotional Arc Points:** {len(data.get('emotional_arc', []))}
                        - **Narrative Pattern:** {data.get('narrative_pattern', 'N/A')}
                        """)

                elif analysis_type == "complexity":
                    data = get_complexity_data(book_id)
                    if data:
                        r = data.get('readability_scores', {})
                        st.markdown(f"""
                        - **Flesch Reading Ease:** {r.get('flesch_reading_ease', 0):.1f}
                        - **Flesch-Kincaid Grade:** {r.get('flesch_kincaid_grade', 0):.1f}
                        - **Word Count:** {data.get('text_statistics', {}).get('word_count', 0):,}
                        - **Unique Words:** {data.get('vocabulary_metrics', {}).get('unique_words', 0):,}
                        """)

                elif analysis_type == "topics":
                    data = get_topics_data(book_id)
                    if data:
                        topics = data.get('topics', [])
                        st.markdown(f"""
                        - **Topics Extracted:** {len(topics)}
                        - **Model Type:** {data.get('model_info', {}).get('model_type', 'N/A')}
                        - **Theme Labels:** {', '.join([t.get('theme_label', '') for t in topics[:3]])}
                        """)

                elif analysis_type == "entities":
                    data = get_entities_data(book_id)
                    if data:
                        chars = data.get('central_characters', [])
                        rels = data.get('relationships', data.get('top_relationships', []))
                        st.markdown(f"""
                        - **Characters Identified:** {len(chars)}
                        - **Relationships Found:** {len(rels)}
                        - **Top Character:** {chars[0].get('name', 'N/A') if chars else 'N/A'}
                        """)

                elif analysis_type == "preprocessing":
                    data = get_preprocessing_data(book_id)
                    if data:
                        st.markdown(f"""
                        - **Token Count:** {data.get('token_count', 0):,}
                        - **Vocabulary Size:** {data.get('vocabulary_size', 0):,}
                        """)


def create_gexf_export(entities_data: Dict[str, Any], title: str) -> str:
    """Create a GEXF format export for network visualization tools."""
    characters = entities_data.get("central_characters", [])
    relationships = entities_data.get("relationships", entities_data.get("top_relationships", []))

    gexf = f'''<?xml version="1.0" encoding="UTF-8"?>
<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2">
  <meta>
    <creator>Literary Analysis Dashboard</creator>
    <description>Character Network for {title}</description>
  </meta>
  <graph defaultedgetype="undirected">
    <nodes>
'''

    for i, char in enumerate(characters):
        name = char.get("name", char.get("entity", f"char_{i}"))
        mentions = char.get("mentions", char.get("count", 1))
        # Escape XML special characters
        name_escaped = name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        gexf += f'      <node id="{i}" label="{name_escaped}">\n'
        gexf += f'        <attvalues>\n'
        gexf += f'          <attvalue for="mentions" value="{mentions}"/>\n'
        gexf += f'        </attvalues>\n'
        gexf += f'      </node>\n'

    gexf += '''    </nodes>
    <edges>
'''

    # Create name to id mapping
    name_to_id = {char.get("name", char.get("entity", "")): i for i, char in enumerate(characters)}

    for j, rel in enumerate(relationships):
        char1 = rel.get("character1", rel.get("entity1", ""))
        char2 = rel.get("character2", rel.get("entity2", ""))
        weight = rel.get("weight", rel.get("co_occurrences", 1))

        id1 = name_to_id.get(char1)
        id2 = name_to_id.get(char2)

        if id1 is not None and id2 is not None:
            gexf += f'      <edge id="{j}" source="{id1}" target="{id2}" weight="{weight}"/>\n'

    gexf += '''    </edges>
  </graph>
</gexf>'''

    return gexf
