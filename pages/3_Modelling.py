
## Streamlit Application for Model Evaluation

import streamlit as st
from PIL import Image
import pandas as pd
import joblib
import numpy as np # linear algebra
import joblib
from streamlit import components
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, PrecisionRecallDisplay,RocCurveDisplay
import config
from sklearn.model_selection import train_test_split # Library to split datset into test and train

def map_categorical_column(df):
    
    ''' Function to map the categorical columns '''
     
    map_payment_type      = {'AA':0, 'AB':1, 'AC':2, 'AD':3, 'AE':4}
    map_employment_status = {'CA':0, 'CB':1, 'CC':2, 'CD':3, 'CE':4,'CF':5,'CG':6}
    map_housing_status    = {'BA':0, 'BB':1, 'BC':2, 'BD':3, 'BE':4,'BF':5,'BG':6}
    map_source            = {'INTERNET':0,'TELEAPP':1}
    map_device_os         = {'windows':0,'other':1,'linux':2,'macintosh':3,'x11':4}
    
    # Updating the mapping in dataframe
    df["payment_type"]                 = df["payment_type"].map(map_payment_type)
    df["employment_status"]            = df["employment_status"].map(map_employment_status)
    df["housing_status"]               = df["housing_status"].map(map_housing_status)
    df["source"]                       = df["source"].map(map_source)
    df["device_os"]                    = df["device_os"].map(map_device_os)

    return df

def set_page_layout():
    
    ''' Function to Create Streamlit Page '''
    
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

    return

def create_sample_set(train_df, non_fraud_sample_sizse):

    ''' Function to split data into train/test '''
                           
    # Fraud Transactions
    train_df_fraud = train_df[train_df.fraud_bool == 1]
    
    # Non Fraud Transactions
    train_df_non_fraud = train_df[train_df.fraud_bool == 0].sample(train_df_fraud.shape[0] * non_fraud_sample_sizse)
    
    # Merge Fraud & Non Fraud
    train_df_merged = pd.concat([train_df_fraud, train_df_non_fraud])
 
    # X & Y
    X                 = train_df_merged.drop(columns=['fraud_bool'])
    y                 = train_df_merged['fraud_bool']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    return X_train, y_train, X_test, y_test

def analyze_model(X_train, y_train, X_test, y_test, usr_sample, usr_model):
    
    # Load Model & Metrics Summary
    classifier = joblib.load(f'./model/sample_{usr_sample}/{usr_model}.pkl')
    # metrics_df = pd.read_csv('./model/results.csv')
    
    # Print the Header Banner
    html_str = f"""
                    <h3 style="background-color:#00A1DE; text-align:center; font-family:arial;color:white">MODEL METRICS: {classifier.__class__.__name__} </h3>
                """

    st.markdown(html_str, unsafe_allow_html=True)
    st.divider()
    
    # Show Graphs
    col1, col2= st.columns(2)
    with col1:
        st.write("Confusion Matrix")
        y_pred = classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                    display_labels=['No Fraud', 'Fraud'])
        disp.plot()
        st.pyplot()

    with col2:
        st.write("Area Under Curve(AUC) - TPR vs FPR")
        disp = RocCurveDisplay.from_predictions(y_test, y_pred)
        disp.plot()
        st.pyplot()

    st.divider()
    
    col3, col4= st.columns(2)
    with col3:
        st.write("Precision Recall Curve")
        disp = PrecisionRecallDisplay.from_predictions(y_test, y_pred)
        disp.plot()
        st.pyplot()
    
    with col4:
        
        if ((usr_model == 'VotingClassifier') or (usr_model == 'StackingClassifier')):
            st.write('Permutation Importance')
            result = permutation_importance(classifier, X_train, y_train)
            perm_importance = pd.DataFrame({'Features':X_train.columns,'Permutation_Importance':result.importances_mean}).sort_values(by='Permutation_Importance', ascending=False).set_index('Features')
            perm_importance.plot(kind="bar")
            st.pyplot()
        else:
            st.write('Feature Importance')
            feature_importances = pd.DataFrame(classifier.feature_importances_,
                                                index = X_train.columns,
                                                columns=['Feature_Importance']).sort_values('Feature_Importance', ascending=False
                                            )
            feature_importances.plot(kind="bar")
            st.pyplot()
    
    return

if __name__ == '__main__':
    
    set_page_layout()

    # Create 3 Columns
    col_data, col_under_sample_strategy, col_model = st.columns(3)
    st.sidebar.title('Analyze')
    with col_data:
        # Load Input Data
        select_file_modelling   =  st.sidebar.selectbox(label = 'Select Data', options=config.file_list)
        input_df                = pd.read_csv(f'./data/{select_file_modelling}')  # Replace with your dataset file

    with col_under_sample_strategy:
        # Load Sampling Strategy
        select_sample_strategy  =  st.sidebar.selectbox(label = 'Fraud vs Non Fraud Sampling', options=config.sampling_strategy)

    with col_model:
        # Load Model
        select_model =  st.sidebar.selectbox(label = 'Model', options=config.classifier_models)

    if st. sidebar.button('Submit'):

        input_df_num = map_categorical_column(input_df)

        X_train, y_train, X_test, y_test = create_sample_set(
                                                                input_df_num[config.reqd_col_modelling], 
                                                                config.sampling_strategy_dict[select_sample_strategy]
                                                            )
        
        analyze_model(X_train, y_train, X_test, y_test, select_sample_strategy, select_model)

    
    