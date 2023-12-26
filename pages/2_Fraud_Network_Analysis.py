
import streamlit as st
from PIL import Image
import pandas as pd     
from pre_process import *
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import networkx as nx
import matplotlib.pyplot as plt
import config

#
def topo_pos(G):
    
    """Display in topological order, with simple offsetting for legibility"""
    
    pos_dict = {}
    for i, node_list in enumerate(nx.topological_generations(G)):
        x_offset = len(node_list) / 2
        y_offset = 0.1
        for j, name in enumerate(node_list):
            pos_dict[name] = (j - x_offset, -i + j * y_offset)

    return pos_dict

#
def  render_network_graph_default(graph_layer, nodes_to_be_removed):
    
    ''' Function to reder graph from networkx package '''

    G=nx.DiGraph()
    G.add_weighted_edges_from(graph_layer)
    
    if nodes_to_be_removed is not None:
       G.remove_nodes_from(nodes_to_be_removed)     
    
    pos = topo_pos(G)
    labels = nx.get_edge_attributes(G,'weight')

    node_color = [G.degree(v) for v in G] # Node Color
    node_size_dict = dict(G.degree)
    node_size  = [node_size_dict[k]*100 for k in node_size_dict] # Node Color

    edge_width = [0.000040 * G[u][v]['weight'] for u, v in G.edges()]  # Edge Width

    fig, ax = plt.subplots(figsize=(6, 6))
    
    nx.draw_networkx(G, 
                    pos,
                    ax=ax,
                    font_size=4,
                    node_color = node_color, 
                    node_size=550,
                    node_shape="o",
                    alpha = 0.7, 
                    width = edge_width, 
                    font_weight="bold",
                    cmap = plt.cm.tab20_r
                    )
    
    nx.draw_networkx_edge_labels(G,
                                pos, 
                                edge_labels=labels,
                                label_pos = 0.6,
                                font_size=3,
                                alpha=1,
                                rotate=False,
                                )
    
    if nodes_to_be_removed is not None:
        ax.set_title("Fraudulent Transaction Count Flow- Customized View")
    else:
        ax.set_title("Fraudulent Transaction Count Flow - Full View")

    fig.tight_layout()
    st.pyplot(fig)
    st.divider()
    
    return

#
def create_hierarchial_network_layer(fraud_df):
    
    ''' Creates Hierarchial Layer for Graph '''

    # Layer 1 for Source (Internet or Teleapp)
    df_layer_1 = fraud_df[['source']]\
                .groupby(['source'])\
                .size()\
                .reset_index()\
                .rename(columns={0:'weight', 'source':'destination'})
    df_layer_1['source'] = 'Source'
    df_layer_1 = df_layer_1[['source', 'destination', 'weight']]
    df_layer_1['weight'] = df_layer_1['weight']

    # Layer 2 Contains Operating Systems
    df_layer_2 = fraud_df[['source', 'device_os']]\
                .groupby(['source', 'device_os'])\
                .size()\
                .reset_index()\
                .rename(columns={0:'weight', 'device_os':'destination'})
    
    # Layer 3 Contains Payment Types
    df_layer_3 = fraud_df[['device_os', 'payment_type']]\
                .groupby(['device_os', 'payment_type'])\
                .size()\
                .reset_index()\
                .rename(columns={0:'weight', 'device_os':'source', 'payment_type':'destination'})
    
    # Layer 4 - If session is alive or not
    df_layer_4 = fraud_df[['payment_type', 'keep_alive_session']]
    df_layer_4['keep_alive_session'] = df_layer_4['keep_alive_session'].map({0:'No', 1:'Yes'})
    df_layer_4 = df_layer_4[['payment_type', 'keep_alive_session']]\
                    .groupby(['payment_type', 'keep_alive_session'])\
                    .size()\
                    .reset_index()\
                    .rename(columns={0:'weight', 'payment_type':'source', 'keep_alive_session':'destination'})
    
    # Layer 4 - How Long the session is alive
    df_layer_5                          = fraud_df[['keep_alive_session', 'session_length_in_minutes']]
    df_layer_5['keep_alive_session']    = df_layer_5['keep_alive_session'].map({0:'No', 1:'Yes'})
    df_layer_5                          = df_layer_5[df_layer_5.session_length_in_minutes > 0]

    df_layer_5['session_length'] = pd.cut(
                            x=df_layer_5['session_length_in_minutes'], 
                            bins=[0, 5, 15, 30, 60, 100],
                            labels=['< 5 Mins', '5-15 mins', '15-30 mins', '30-60 mins', '>60 mins']
                            )

    df_layer_5 = df_layer_5[['keep_alive_session', 'session_length']]\
                    .groupby(['keep_alive_session', 'session_length'])\
                    .size()\
                    .reset_index()\
                    .rename(columns={0:'weight', 'keep_alive_session':'source', 'session_length':'destination'})
    
    # All Connected layer datframe
    graph_df = pd.concat([df_layer_1, df_layer_2, df_layer_3, df_layer_4, df_layer_5])

    # Tuple containing Start, Destination & Weight
    # [('Source', 'INTERNET', 10917), ('Source', 'TELEAPP', 112)]

    graph_layer = list(graph_df[['source', 'destination', 'weight']].apply(tuple, axis=1))

    return graph_df, graph_layer

def streamlit_interface():
    
    # Page Setup
    st.set_page_config(
                        page_title="IntelliFraud",
                        layout="wide",
                        initial_sidebar_state="auto",
                        page_icon='./images/fraud-detection.png',
                    )
    # Hide Footer Setup
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    # Condense Layout
    padding = 0
    st.markdown(f""" <style>
        .reportview-container .main .block-container{{
            padding-top: {padding}rem;
            padding-right: {padding}rem;
            padding-left: {padding}rem;
            padding-bottom: {padding}rem;
        }} </style> """, unsafe_allow_html=True)

    # Header
    image = Image.open("./images/intellifraud_icon.png")
    st.image(image, width = 300)
    
    # Print the Header Banner
    html_str = f"""
                    <h3 style="background-color:#00A1DE; text-align:center; font-family:arial;color:white">FRAUD DETECTION SYSTEM </h3>
                """

    st.markdown(html_str, unsafe_allow_html=True)
    st.divider()
    
    # Read Dataset
    file_list = config.file_list
    intellifraud_dataset = pd.DataFrame()

    # Sidebar Analysis
    st.sidebar.title('Analyze')
    
    # Load Input Data
    select_data   =  st.sidebar.selectbox(label = 'Select Data', options=config.file_list)

    nodes_to_be_removed     =  st.sidebar.multiselect(label = 'Remove Nodes', options=config.remove_nodes)
    
    if st. sidebar.button('Submit'):

        # Read Dataset
        input_df_fraud = pd.read_csv(f"./data/{select_data}")

        # Extract Fraud Transactions
        intellifraud_dataset = input_df_fraud[input_df_fraud.fraud_bool==1]
        
        # Create hierarchial Graph Layer
        _, graph_layer = create_hierarchial_network_layer(intellifraud_dataset)
        render_network_graph_default(graph_layer, None)

        # Remove nodes
        if len(nodes_to_be_removed) > 0:
            print(nodes_to_be_removed)
            render_network_graph_default(graph_layer, nodes_to_be_removed)

    return

if __name__ == '__main__':
    
    # Streamlit Inteface
    streamlit_interface()

    
    