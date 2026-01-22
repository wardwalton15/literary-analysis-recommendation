"""Network visualization component for character relationships."""

from typing import Dict, Any, Optional
import streamlit.components.v1 as components


def create_character_network_html(network_data: Dict[str, Any], height: int = 600) -> str:
    """
    Create an interactive character network visualization using PyVis.

    Args:
        network_data: Dictionary containing characters and relationships
        height: Height of the visualization in pixels

    Returns:
        HTML string for the network visualization
    """
    try:
        from pyvis.network import Network
    except ImportError:
        return "<p>PyVis not installed. Run: pip install pyvis</p>"

    if not network_data:
        return "<p>No network data available.</p>"

    # Create network
    net = Network(
        height=f"{height}px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#333333",
        directed=False
    )

    # Configure physics for better layout
    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "stabilization": {
                "enabled": true,
                "iterations": 200
            },
            "barnesHut": {
                "gravitationalConstant": -2000,
                "centralGravity": 0.3,
                "springLength": 150,
                "springConstant": 0.04
            }
        },
        "nodes": {
            "font": {
                "size": 14,
                "face": "Arial"
            }
        },
        "edges": {
            "color": {
                "inherit": false
            },
            "smooth": {
                "enabled": true,
                "type": "continuous"
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": true
        }
    }
    """)

    # Color palette for communities
    community_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    # Add nodes (characters)
    characters = network_data.get("central_characters", [])
    character_info = {c.get("name", c.get("entity", "")): c for c in characters}

    # Get max mentions for scaling
    max_mentions = max((c.get("mentions", c.get("count", 1)) for c in characters), default=1)

    for char in characters:
        name = char.get("name", char.get("entity", "Unknown"))
        mentions = char.get("mentions", char.get("count", 1))
        centrality = char.get("degree_centrality", 0.5)
        community = char.get("community", 0)

        # Scale node size based on mentions
        size = 15 + (mentions / max_mentions) * 35

        # Get community color
        color = community_colors[community % len(community_colors)]

        # Create hover title
        title = f"""
        <b>{name}</b><br>
        Mentions: {mentions}<br>
        Centrality: {centrality:.3f}<br>
        Community: {community}
        """

        net.add_node(
            name,
            label=name,
            size=size,
            color=color,
            title=title,
            borderWidth=2,
            borderWidthSelected=4
        )

    # Add edges (relationships)
    relationships = network_data.get("relationships", network_data.get("top_relationships", []))

    # Get max weight for scaling
    max_weight = max((r.get("weight", r.get("co_occurrences", 1)) for r in relationships), default=1)

    for rel in relationships:
        char1 = rel.get("character1", rel.get("entity1", ""))
        char2 = rel.get("character2", rel.get("entity2", ""))
        weight = rel.get("weight", rel.get("co_occurrences", 1))

        # Only add edge if both nodes exist
        if char1 in character_info and char2 in character_info:
            # Scale edge width
            width = 1 + (weight / max_weight) * 5

            # Edge color based on weight
            opacity = min(0.3 + (weight / max_weight) * 0.7, 1.0)
            edge_color = f"rgba(100, 100, 100, {opacity})"

            title = f"{char1} ↔ {char2}<br>Co-occurrences: {weight}"

            net.add_edge(
                char1,
                char2,
                value=weight,
                width=width,
                color=edge_color,
                title=title
            )

    # Generate HTML
    html = net.generate_html()

    return html


def render_network_in_streamlit(network_data: Dict[str, Any], height: int = 600) -> None:
    """
    Render the character network in Streamlit.

    Args:
        network_data: Dictionary containing characters and relationships
        height: Height of the visualization in pixels
    """
    html = create_character_network_html(network_data, height)
    components.html(html, height=height + 50)


def create_simple_network_stats(network_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate basic network statistics.

    Args:
        network_data: Dictionary containing characters and relationships

    Returns:
        Dictionary with network statistics
    """
    characters = network_data.get("central_characters", [])
    relationships = network_data.get("relationships", network_data.get("top_relationships", []))

    num_nodes = len(characters)
    num_edges = len(relationships)

    # Calculate density (for undirected graph)
    max_edges = (num_nodes * (num_nodes - 1)) / 2 if num_nodes > 1 else 1
    density = num_edges / max_edges if max_edges > 0 else 0

    # Count unique communities
    communities = set(c.get("community", 0) for c in characters)

    # Find most central character
    most_central = max(characters, key=lambda x: x.get("degree_centrality", 0)) if characters else None

    return {
        "total_characters": num_nodes,
        "total_relationships": num_edges,
        "network_density": density,
        "num_communities": len(communities),
        "most_central": most_central.get("name", "N/A") if most_central else "N/A",
        "most_central_value": most_central.get("degree_centrality", 0) if most_central else 0,
    }
