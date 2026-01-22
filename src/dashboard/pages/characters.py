"""Character Network view page."""

import streamlit as st
from typing import Dict, Any
import pandas as pd

from src.dashboard.utils.data_loader import get_entities_data
from src.dashboard.components.network import (
    render_network_in_streamlit,
    create_simple_network_stats,
)
from src.dashboard.components.charts import create_bar_chart


def render_characters_page(sidebar_state: Dict[str, Any]) -> None:
    """Render the character network page."""

    book_id = sidebar_state.get("book_id")
    book_title = sidebar_state.get("book_title", "Selected Work")

    st.header(f"🔗 Character Network: {book_title}")

    if not book_id:
        st.warning("Please select a book from the sidebar.")
        return

    # Load entities data
    entities_data = get_entities_data(book_id)

    if not entities_data:
        st.error("No character/entity data available for this book.")
        return

    # Extract network data from the entities structure
    character_network = entities_data.get("character_network", {})
    characters_list = entities_data.get("characters", [])

    # Section 1: Network Statistics
    st.subheader("📊 Network Statistics")

    # Use character_network data for stats
    network_stats = {
        "total_characters": len(characters_list),
        "total_relationships": len(entities_data.get("relationships", [])),
        "network_density": character_network.get("density", 0),
        "num_communities": len(character_network.get("communities", [])),
        "modularity": character_network.get("modularity", 0),
    }

    # Find most central character
    central_chars = character_network.get("central_characters", [])
    if central_chars:
        most_central = central_chars[0]
        network_stats["most_central"] = most_central.get("name", most_central.get("text", "N/A"))
        network_stats["most_central_value"] = most_central.get("degree_centrality", 0)
    else:
        # Fallback to character list
        if characters_list:
            most_central = max(characters_list, key=lambda x: x.get("count", 0))
            network_stats["most_central"] = most_central.get("text", "N/A")
            network_stats["most_central_value"] = most_central.get("count", 0) / 100
        else:
            network_stats["most_central"] = "N/A"
            network_stats["most_central_value"] = 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Characters",
            network_stats["total_characters"],
            help="Number of named entities identified"
        )

    with col2:
        st.metric(
            "Total Relationships",
            network_stats["total_relationships"],
            help="Number of character co-occurrences"
        )

    with col3:
        st.metric(
            "Network Density",
            f"{network_stats['network_density']:.3f}",
            help="Ratio of actual to possible connections"
        )

    with col4:
        st.metric(
            "Communities",
            network_stats["num_communities"],
            help="Detected character clusters"
        )

    # Most central character highlight
    st.info(f"**Most Central Character:** {network_stats['most_central']} (centrality: {network_stats['most_central_value']:.3f})")

    st.markdown("---")

    # Section 2: Interactive Network Graph
    st.subheader("🕸️ Interactive Character Network")

    st.markdown("""
    *Hover over nodes for character details. Node size represents mention frequency.
    Edge thickness shows relationship strength (co-occurrences). Colors indicate communities.*
    """)

    # Check if PyVis is available
    try:
        # Pass the network data in the format expected by the component
        network_data = {
            "central_characters": central_chars if central_chars else [
                {"name": c.get("text", ""), "mentions": c.get("count", 0), "degree_centrality": 0.5, "community": 0}
                for c in characters_list[:20]
            ],
            "relationships": entities_data.get("relationships", [])
        }
        render_network_in_streamlit(network_data, height=600)
    except Exception as e:
        st.error(f"Could not render network visualization: {e}")
        st.info("Install PyVis with: `pip install pyvis`")

    st.markdown("---")

    # Section 3: Central Characters Table
    st.subheader("👥 Central Characters")

    # Use central_characters from network if available, otherwise use characters list
    characters = central_chars if central_chars else characters_list

    if characters:
        # Create DataFrame for display - handle both data formats
        def get_char_name(c):
            return c.get("name", c.get("text", c.get("entity", "Unknown")))

        def get_char_mentions(c):
            return c.get("mentions", c.get("count", 0))

        def get_char_centrality(c):
            return c.get("degree_centrality", 0)

        sorted_chars = sorted(characters, key=lambda x: get_char_centrality(x) or get_char_mentions(x), reverse=True)[:20]

        char_df = pd.DataFrame([
            {
                "Rank": i + 1,
                "Name": get_char_name(c),
                "Mentions": get_char_mentions(c),
                "Degree Centrality": f"{get_char_centrality(c):.4f}",
                "Betweenness": f"{c.get('betweenness_centrality', 0):.4f}",
                "Community": c.get("community", 0),
            }
            for i, c in enumerate(sorted_chars)
        ])

        # Display with filtering
        st.dataframe(
            char_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", width="small"),
                "Name": st.column_config.TextColumn("Character", width="medium"),
                "Mentions": st.column_config.NumberColumn("Mentions", width="small"),
                "Degree Centrality": st.column_config.TextColumn("Degree", width="small"),
                "Betweenness": st.column_config.TextColumn("Betweenness", width="small"),
                "Community": st.column_config.NumberColumn("Community", width="small"),
            }
        )

        # Bar chart of top characters by mentions
        top_chars = sorted(characters, key=lambda x: get_char_mentions(x), reverse=True)[:10]
        names = [get_char_name(c) for c in top_chars]
        mentions = [get_char_mentions(c) for c in top_chars]

        fig = create_bar_chart(
            x=names,
            y=mentions,
            title="Top 10 Characters by Mention Count",
            x_label="Character",
            y_label="Mentions",
            color="#9467bd"
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No character data available.")

    st.markdown("---")

    # Section 4: Community Analysis
    st.subheader("🏘️ Community Analysis")

    # Use communities from network data if available
    network_communities = character_network.get("communities", [])

    if network_communities:
        # Communities are already grouped
        for i, community_members in enumerate(network_communities):
            label = f"Community {i+1}"

            with st.expander(f"**{label}** ({len(community_members)} members)", expanded=(i < 2)):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown("**Members:**")
                    for name in community_members[:10]:
                        st.markdown(f"- {name}")
                    if len(community_members) > 10:
                        st.caption(f"...and {len(community_members) - 10} more")

                with col2:
                    st.markdown("**Summary:**")
                    st.markdown(f"- Members: {len(community_members)}")
    else:
        # Fallback: Group characters by community field
        communities = {}
        for char in characters:
            comm = char.get("community", 0)
            if comm not in communities:
                communities[comm] = []
            communities[comm].append(char)

        if communities:
            # Community descriptions (can be enhanced with interpretation)
            community_labels = {
                0: "Primary Characters",
                1: "Secondary Group",
                2: "Tertiary Group",
                3: "Minor Characters",
            }

            for comm_id in sorted(communities.keys()):
                members = communities[comm_id]
                label = community_labels.get(comm_id, f"Community {comm_id}")

                with st.expander(f"**{label}** ({len(members)} members)", expanded=(comm_id < 2)):
                    # List members
                    member_names = [get_char_name(m) for m in members]
                    member_mentions = [get_char_mentions(m) for m in members]

                    # Sort by mentions
                    sorted_members = sorted(zip(member_names, member_mentions), key=lambda x: x[1], reverse=True)

                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown("**Members:**")
                        for name, mention_count in sorted_members[:10]:
                            st.markdown(f"- {name} ({mention_count} mentions)")
                        if len(sorted_members) > 10:
                            st.caption(f"...and {len(sorted_members) - 10} more")

                    with col2:
                        st.markdown("**Summary:**")
                        total_mentions = sum(m for _, m in sorted_members)
                        avg_mentions = total_mentions / len(sorted_members) if sorted_members else 0
                        st.markdown(f"""
                        - Total mentions: {total_mentions:,}
                        - Avg mentions: {avg_mentions:.1f}
                        """)

    st.markdown("---")

    # Section 5: Top Relationships
    st.subheader("🤝 Top Relationships")

    relationships = entities_data.get("relationships", entities_data.get("top_relationships", []))

    if relationships:
        # Sort by weight/co-occurrences
        sorted_rels = sorted(
            relationships,
            key=lambda x: x.get("weight", x.get("co_occurrences", 0)),
            reverse=True
        )[:15]

        rel_df = pd.DataFrame([
            {
                "Character 1": r.get("character1", r.get("entity1", "")),
                "Character 2": r.get("character2", r.get("entity2", "")),
                "Co-occurrences": r.get("weight", r.get("co_occurrences", 0)),
                "Context": r.get("context", "")[:100] + "..." if r.get("context", "") else "N/A"
            }
            for r in sorted_rels
        ])

        st.dataframe(rel_df, hide_index=True, use_container_width=True)

        # Sample contexts
        st.markdown("**Sample Relationship Contexts:**")

        for rel in sorted_rels[:3]:
            char1 = rel.get("character1", rel.get("entity1", ""))
            char2 = rel.get("character2", rel.get("entity2", ""))
            contexts = rel.get("sample_contexts", rel.get("contexts", []))

            if contexts:
                with st.expander(f"{char1} ↔ {char2}"):
                    for ctx in contexts[:2]:
                        st.markdown(f"> *\"{ctx[:200]}...\"*" if len(ctx) > 200 else f"> *\"{ctx}\"*")

    else:
        st.info("No relationship data available.")
