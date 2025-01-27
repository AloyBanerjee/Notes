import streamlit as st
import os
from typing import List, Dict
from neo4j import GraphDatabase
from langchain_groq import ChatGroq
#from langchain.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
#from langchain.vectorstores import Neo4jVector
from langchain_community.vectorstores import Neo4jVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import networkx as nx
from pyvis.network import Network
import tempfile
import plotly.graph_objects as go
# from streamlit_plotly import plotly_chart
from streamlit import plotly_chart
from neo4j import GraphDatabase
from pyvis.network import Network
import streamlit.components.v1 as components

import os 
from dotenv import load_dotenv

load_dotenv()

import base64
import re

# Streamlit page configuration
st.set_page_config(layout="wide", page_title="Knowledge Graph RAG System")

# Custom CSS for better UI
st.markdown("""
<style>
    
    .graph-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="mixtral-8x7b-32768"
)

# Neo4j Configuration
NEO4J_URI = "neo4j+s://eddeca98.databases.neo4j.io"
NEO4J_USER = "no4j"
NEO4J_PASSWORD = "2_Yy9ckQrsLGzxgyta-NtJ6Sl_LihqHXdjJQMM"

class KnowledgeGraphRAG:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        # self.embeddings = HuggingFaceEmbeddings(
        #     model_name="sentence-transformers/all-mpnet-base-v2"
        # )
        self.embeddings = OllamaEmbeddings()
    
    def delete_database(self):
        """Delete all nodes and relationships in the database"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            try:
                session.run("CALL db.index.vector.drop('document_vectors')")
            except Exception as e:
                st.warning("Vector index might not exist or was already deleted.")
            
    def create_vector_store(self, documents: List):
        """Create vector store in Neo4j"""
        vector_store = Neo4jVector.from_documents(
            documents,
            self.embeddings,
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            index_name="document_vectors",
            node_label="Document",
            embedding_node_property="embedding",
            text_node_property="text"
        )
        return vector_store

    def _parse_relationships(self, llm_response: str) -> List[Dict]:
        """Parse LLM relationship extraction response"""
        relationships = []
        # Regular expression pattern to match the relationship format
        #pattern = r'\(([^)]+)\)-\[([^\]]+)\]->\(([^)]+)\)'
        pattern = r'\(\s*([^)]+?)\s*\)-\[\s*([^\]]+?)\s*\]->\(\s*([^)]+?)\s*\)'
        # Split the response into lines and process each line
        for line in llm_response.split('\n'):
            line = line.strip()
            matches = re.findall(pattern, line)
            
            for match in matches:
                if len(match) == 3:  # Ensure we have all three components
                    entity1, relationship, entity2 = match
                    # Clean up the extracted texts
                    entity1 = entity1.strip()
                    relationship = relationship.strip()
                    entity2 = entity2.strip()
                    
                    if entity1 and relationship and entity2:  # Ensure none are empty
                        relationships.append({
                            'entity1': entity1,
                            'relationship': relationship,
                            'entity2': entity2
                        })
        
        return relationships

    def create_knowledge_graph(self, documents: List):
        """Extract entities and relationships to create knowledge graph"""
        with self.driver.session() as session:
            for doc in documents:
                prompt = f"""
                Extract key entities and their relationships from this text. 
                Format each relationship exactly as: (entity1)-[relationship]->(entity2)
                Return one relationship per line.
                Only include clear, explicit relationships from the text.
                Text: {doc.page_content}
                """
                response = llm.predict(prompt)
                
                relationships = self._parse_relationships(response)
                for rel in relationships:
                    if all(rel.values()):  # Check that no values are empty
                        session.run("""
                        MERGE (e1:Entity {name: $entity1})
                        MERGE (e2:Entity {name: $entity2})
                        MERGE (e1)-[:RELATES {type: $relationship}]->(e2)
                        """, rel)

    def create_3d_graph(self):
        """Generate 3D graph visualization using Plotly"""
        G = nx.DiGraph()
        
        with self.driver.session() as session:
            result = session.run("""
            MATCH (e1:Entity)-[r:RELATES]->(e2:Entity)
            RETURN e1.name as source, r.type as relationship, e2.name as target
            """)
            records = list(result)
            
        if not records:  # If no relationships exist
            return None
            
        for record in records:
            G.add_edge(record["source"], record["target"])
            
        pos = nx.spring_layout(G, dim=3)
        
        edge_x = []
        edge_y = []
        edge_z = []
        for edge in G.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])

        edges_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        node_z = []
        for node in G.nodes():
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)

        nodes_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            hovertext=list(G.nodes()),
            hoverinfo='text',
            marker=dict(
                size=8,
                color='#00ff00',
                line_width=2))

        fig = go.Figure(data=[edges_trace, nodes_trace])
        fig.update_layout(
            showlegend=False,
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig

    def fetch_graph_data(self, query, question=None):
        with self.driver.session() as session:
            if question:
                result = session.run(query, question=question)
            else:
                result = session.run(query)
            return [record.data() for record in result]
    
    def query(self, question: str) -> Dict:
        """Query both vector store and knowledge graph"""
        vector_store = Neo4jVector(
            self.embeddings,
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            index_name="document_vectors"
        )
        retriever = vector_store.as_retriever()
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )
        
        rag_answer = qa_chain.invoke(question)
        
        kg_data = []

        with self.driver.session() as session:
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
                kg_data = self.fetch_graph_data(query, question=question)
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
                kg_data = self.fetch_graph_data(query)
        #st.sidebar.info(query)   
        return {
            "rag_answer": rag_answer,
            "knowledge_graph": kg_data
        }

    def visualize_graph(self, data):
        # Create the network with a white background
        net = Network(height="750px", width="100%", bgcolor="white", font_color="black")

        # Enable physics for a better layout
        net.barnes_hut()

        for record in data:
            source = record["source"]
            target = record["target"]
            relationship = record['relationship_properties'].get('type', 'N/A')#record["relationship"]
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
 
    
def main():
    st.title("🌟 Advanced RAG System with Knowledge Graph")
    
    # Initialize system
    rag_system = KnowledgeGraphRAG()
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Controls")
        if st.button("🗑️ Delete Database", help="Clear all data and start fresh"):
            with st.spinner("Deleting database..."):
                rag_system.delete_database()
            st.success("Database cleared successfully!")       
        
        st.markdown("### 📊 Relationship Information")
        
    
    # Main content area    
    col1, col2 = st.tabs(["📄 Upload Document", "🔍 Query"])

    with col1:
        st.markdown("### 📄 Document Upload")
        uploaded_file = st.file_uploader(
            "Drop your PDF document here",
            type="pdf",
            help="Upload a PDF document to analyze"
        )
        
        if uploaded_file:
            with st.spinner("🔄 Processing document..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    tmp.write(uploaded_file.getvalue())
                    loader = PyPDFLoader(tmp.name)
                    documents = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                splits = text_splitter.split_documents(documents)
                
                rag_system.create_vector_store(splits)
                rag_system.create_knowledge_graph(splits)

            st.success("✅ Document processed successfully!")

            #3D Graph Visualization
            st.markdown("### 🌐 Knowledge Graph Visualization")    
            with st.container():
                query = """
                    MATCH (e1:Entity)-[r:RELATES]->(e2:Entity)
                    RETURN 
                        e1.name AS source, 
                        type(r) AS relationship, 
                        properties(r) AS relationship_properties, 
                        e2.name AS target
                    LIMIT 50
                    """
                graph_data = rag_system.fetch_graph_data(query)
                st.session_state.graph_data = graph_data
                st.sidebar.success(f"Found {len(graph_data)} relationships!")
                st.sidebar.info(graph_data)
                rag_system.visualize_graph(graph_data)

                # Render the graph in Streamlit
                HtmlFile = open("interactive_graph.html", "r", encoding="utf-8")
                components.html(HtmlFile.read(), height=1000)
                
    with col2:
        st.markdown("### 🔍 Query Interface")
        question = st.text_input("Ask a question:", placeholder="Type your question here...")
    
        results = {
                "rag_answer": "",
                "knowledge_graph": ""
            }    
        if question:
            with st.spinner("🤔 Analyzing..."):
                results = rag_system.query(question)
            
            st.markdown("### 📝 Answer")
                
            st.markdown(f"<pre style='background-color: #f4f4f4; padding: 10px;'>{results['rag_answer']['result']}</pre>", unsafe_allow_html=True)

            
            #st.sidebar.write(results)
            if results["knowledge_graph"]:
                st.markdown("### 🔗 Knowledge Graph Connections")
                st.markdown(f"Found {len(results['knowledge_graph'])} relationships:")
                for rel in results["knowledge_graph"]:
                    st.markdown(f"🔹 {rel['source']} → *{rel['relationship']}({rel['relationship_properties'].get('type', 'N/A')})* → {rel['target']}")
        
            if st.session_state.graph_data:
                # Extract unique relationship types for filtering
                relationship_types = list(set(
                    rel['relationship_properties'].get('type', 'N/A')
                    for rel in st.session_state.graph_data
                ))

                # Add a multiselect widget
                selected_types = st.sidebar.multiselect(
                    "Filter by Relationship Type:",
                    options=relationship_types,
                    default=relationship_types,  # Default: Show all
                )

                # Filter the graph data
                filtered_graph_data = [
                    rel for rel in st.session_state.graph_data
                    if rel['relationship_properties'].get('type', 'N/A') in selected_types
                ]
            else:
                filtered_graph_data = []

            st.markdown("### 🌐 Knowledge Graph Visualization based on Query")

            if filtered_graph_data:
                rag_system.visualize_graph(filtered_graph_data)

                # Render the graph in Streamlit
                HtmlFile = open("interactive_graph.html", "r", encoding="utf-8")
                components.html(HtmlFile.read(), height=800)
            else:
                st.warning("No relationships match the selected filters!")

if __name__ == "__main__":
    main()