import streamlit as st
from neo4j import GraphDatabase
from pyvis.network import Network
import streamlit.components.v1 as components

# Neo4j connection setup
NEO4J_URI = "neo4j+s://eddeca98.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "2_Yy9ckQrsLGzxgtLm4LlUa-NtJ6Sl_LihqHXdjJQMM"

# Create Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Function to fetch graph data from Neo4j
def fetch_graph_data(query, question=None):
    with driver.session() as session:
        if question:
            result = session.run(query, question=question)
        else:
            result = session.run(query)
        return [record.data() for record in result]

# Function to create an enhanced interactive graph using pyvis
def visualize_graph(data):
    net = Network(height="750px", width="100%", bgcolor="white", font_color="black")

    # Enable physics for better layout
    net.barnes_hut()

    # Add nodes and edges
    for record in data:
        source = record["source"]
        target = record["target"]
        relationship = record["relationship"]
        properties = record.get("relationship_properties", {})

        # Convert properties dictionary into a formatted string for tooltip
        if isinstance(properties, dict):  # Ensure it's a dictionary
            property_text = "<br>".join([f"{key}: {value}" for key, value in properties.items()])
        else:
            property_text = "No additional properties"
        
        # Add nodes with hover info
        net.add_node(source, label=source, title=f"Entity: {source}", color="#ffcc00", shape="dot")
        net.add_node(target, label=target, title=f"Entity: {target}", color="#00ccff", shape="box")

        # Add edges with hover info
        #net.add_edge(source, target, title=relationship, color="lightgray")
        net.add_edge(
            source, 
            target, 
            title=f"Relationship: {relationship}<br>{property_text}",  # Hover tooltip
            #label=relationship,  # Show relationship type on the edge
            color="lightgray"
        )

    # Add options for more interactivity
    net.set_options("""
    const options = {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -2000,
          "centralGravity": 0.3,
          "springLength": 95,
          "springConstant": 0.04,
          "damping": 0.09
        }
      },
      "edges": {
        "color": {
          "inherit": true
        },
        "smooth": {
          "type": "dynamic"
        }
      },
      "interaction": {
        "hover": true,
        "multiselect": true,
        "dragNodes": true
      }
    }
    """)

    # Save the graph to an HTML file
    net.save_graph("interactive_graph.html")


    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")

    # Enable physics for better layout
    net.barnes_hut()

    # Add nodes and edges with relationship name
    for record in data:
        source = record["source"]
        target = record["target"]
        relationship = record["relationship"]

        # Add nodes with hover info
        net.add_node(source, label=source, title=f"Entity: {source}", color="#ffcc00", shape="dot")
        net.add_node(target, label=target, title=f"Entity: {target}", color="#00ccff", shape="dot")

        # Add edges with hover and label
        net.add_edge(
            source, 
            target, 
            title=f"Relationship: {relationship}",  # Tooltip on hover
            label=relationship,  # Relationship name shown on edge
            color="lightgray"
        )

    # Add options for more interactivity
    net.set_options("""
    const options = {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -2000,
          "centralGravity": 0.3,
          "springLength": 95,
          "springConstant": 0.04,
          "damping": 0.09
        }
      },
      "edges": {
        "color": {
          "inherit": true
        },
        "smooth": {
          "type": "dynamic"
        }
      },
      "interaction": {
        "hover": true,
        "multiselect": true,
        "dragNodes": true
      }
    }
    """)

    # Save the graph to an HTML file
    net.save_graph("interactive_graph.html")

def visualize_graph_updated(data):
    # Create the network with a white background
    net = Network(height="750px", width="100%", bgcolor="white", font_color="black")

    # Enable physics for a better layout
    net.barnes_hut()

    for record in data:
        source = record["source"]
        target = record["target"]
        relationship = record["relationship"]
        properties = record.get("relationship_properties", {})

        # Convert properties dictionary into a formatted string for tooltips       
        if isinstance(properties, dict):  # Ensure it's a dictionary
            property_text = "<br>".join([f"{key}: {value}" for key, value in properties.items()])
        else:
            property_text = "No additional properties"

        # Add nodes with colors that contrast well on white
        net.add_node(source, label=source, title=f"Entity: {source}", color="#1f77b4", shape="box")  # Blue node
        net.add_node(target, label=target, title=f"Entity: {target}", color="#ff7f0e", shape="box")  # Orange node

        # Add edges with black color and labels for relationships
        net.add_edge(
            source, 
            target, 
            title=f"Relationship: {relationship}<br>{property_text}",  # Hover tooltip
            label=relationship,  # Relationship type shown on edge
            color="gray"
        )

    # Add custom options for interactivity and styling
    net.set_options("""
    {
    "physics": {
        "enabled": true,
        "barnesHut": {
        "gravitationalConstant": -2000,
        "centralGravity": 0.3,
        "springLength": 95,
        "springConstant": 0.04,
        "damping": 0.09
        }
    },
    "nodes": {
        "font": {
        "color": "black",
        "size": 14
        },
        "borderWidth": 2
    },
    "edges": {
        "color": {
        "inherit": false,
        "color": "black"
        },
        "font": {
        "color": "black",
        "size": 12,
        "align": "top"
        },
        "smooth": {
        "type": "dynamic"
        }
    },
    "interaction": {
        "hover": true,
        "dragNodes": true,
        "zoomView": true
    }
    }
    """)


    # Save the graph to an HTML file
    net.save_graph("interactive_graph_white.html")

# Streamlit App
st.title("Enhanced Neo4j Graph Visualization")
st.sidebar.header("Configuration")

# Input question for filtering
question = st.sidebar.text_input("Enter a keyword to filter entities:", "")

# Query Neo4j
st.sidebar.write("Fetching data...")
if question:
    query = """
    MATCH (e1:Entity)-[r:RELATES]->(e2:Entity)
    WHERE e1.name CONTAINS $question OR e2.name CONTAINS $question
    RETURN 
        e1.name AS source, 
        type(r) AS relationship, 
        properties(r) AS relationship_properties, 
        e2.name AS target
    LIMIT 50
    """
    graph_data = fetch_graph_data(query, question=question)
else:
    query = """
    MATCH (e1:Entity)-[r:RELATES]->(e2:Entity)
    RETURN 
        e1.name AS source, 
        type(r) AS relationship, 
        properties(r) AS relationship_properties, 
        e2.name AS target
    LIMIT 50
    """
    graph_data = fetch_graph_data(query)


# Check if data is retrieved
if graph_data:
    st.sidebar.success(f"Found {len(graph_data)} relationships!")
    st.sidebar.info(graph_data)
    st.write("### Interactive Graph Visualization")
    visualize_graph_updated(graph_data)

    # Render the graph in Streamlit
    HtmlFile = open("interactive_graph.html", "r", encoding="utf-8")
    components.html(HtmlFile.read(), height=800)
else:
    st.sidebar.error("No data found!")
    st.write("No graph data to display. Try changing your search criteria.")

# Close the Neo4j driver
driver.close()
