import streamlit as st
from PIL import Image

# Page Setup
st.set_page_config(
                    page_title="IntelliFraud",
                    layout="centered",
                    initial_sidebar_state="auto",
                    page_icon='./images/fraud-detection.png',
                )
st.set_option('deprecation.showPyplotGlobalUse', False)

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

st.write("## IntelliFraud  - Account Opening Fraud Detection")

# st.sidebar.success("Select a demo above.")
st.info('Goal')
st.markdown(
    """

IntelliFraud represents an advanced online tool to revolutionize the identification of fraudulent applications for bank account openings. The tool incorporates an intuitive interactive dashboard that facilitates users in visually exploring extensive statistical data derived from all submitted applications.

What distinguishes IntelliFraud is its distinctive methodology for scrutinizing fraud transactions through interactive graph network analysis , uses Voting & Stacking Classifer modelling technique to detect fraud and analyzes key features impacting the fraud vs non fraud decision using SHAP & ELI5. 

Tool is divided into 4 sections:

1. EDA - User can choose the dataset (Currently 6 Variants of Account Opening Fraud, customizable to any datasets) and perform exploratory data analysis.

2. Fraud Network Analysis - User can choose the dataset and analyze the transaction thru Graph Network Analysis by removing or adding nodes/features to understand the flow of fraudulent transactions thru the network.

3. Modelling - User can choose the data, Fraud vs Non Fraud Sampling Method and Model to observe and analyze various classification metrices and choose the apt model.

4. Inference - Interactive User Interface for users to analyze the account opening fraud by adding data to the features and choosing the model. Interface also enables user to understand the factors leading to the model decision.

""")

st.info('Visualizations')

col1, col2= st.columns(2)
with col1:
    image = Image.open("./images/eda_1.jpg").resize((350, 300))
    st.image(image, width = 300)
    st.divider()
    
    image = Image.open("./images/Fraudulent_Transaction_Flow_2.jpg").resize((350, 300))
    st.image(image, width = 300)
    st.divider()

    image = Image.open("./images/modelling_2.jpg").resize((350, 300))
    st.image(image, width = 300)

with col2:
    image = Image.open("./images/eda_2.jpg").resize((350, 300))
    st.image(image, width = 300)
    st.divider()

    image = Image.open("./images/modelling.jpg").resize((350, 300))
    st.image(image, width = 300)

st.divider()
image = Image.open("./images/inference.jpg")
st.image(image, width = 600)

st.info('Code Repository')
st.markdown("""Full Code : https://github.com/jbanerje/dva_cse6242_intellifraud
            . Please folow the readme documentation for installations""")
