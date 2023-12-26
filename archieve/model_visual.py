
## Streamlit Application for Model Evaluation

import streamlit as st
from PIL import Image
import pandas as pd
import joblib
import numpy as np # linear algebra
import config
import joblib
import eli5
from eli5.sklearn import PermutationImportance
from streamlit import components
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, PrecisionRecallDisplay,RocCurveDisplay

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

def create_sample_set(train_df, non_fraud_sample_sizse):
    
    ''' Function to Create Data for Modelling '''
    
    # Select columns
    train_df = train_df[['prev_address_months_count', 'date_of_birth_distinct_emails_4w','credit_risk_score', 'bank_months_count', 
                         'proposed_credit_limit','customer_age', 'housing_status','device_os', 'employment_status',
                         'keep_alive_session','has_other_cards','phone_home_valid','payment_type', 'fraud_bool', 'month']]
    
    # Fraud Transactions
    train_df_fraud = train_df[train_df.fraud_bool == 1]
    
    # Non Fraud Transactions
    train_df_non_fraud = train_df[train_df.fraud_bool == 0].sample(train_df_fraud.shape[0] * non_fraud_sample_sizse)
    
    # Merge Fraud & Non Fraud
    train_df_merged = pd.concat([train_df_fraud, train_df_non_fraud])

    # Shuffle
    train_df_merged.iloc[:,:] = train_df_merged.sample(frac=1,random_state=123,ignore_index=True)
    
    # X & Y
    X                 = train_df_merged.drop(columns=['fraud_bool'])
    X['customer_age'] = X['customer_age'].apply(lambda x: 0 if x < 50 else 1)
    y                 = train_df_merged[['fraud_bool', 'month']]
    
    # Train Dataframe
    X_train = X[X.month <= 6].drop(columns=['month'])
    y_train = y[y.month <= 6].drop(columns=['month']).values.ravel()

    # Test Dataframe
    X_test = X[X.month > 6].drop(columns=['month'])
    y_test = y[y.month > 6].drop(columns=['month']).values.ravel()

    return X_train, y_train, X_test, y_test

def streamlit_interface(X_train, y_train, X_test, y_test):
    
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
    
    # Select Model
    usr_model = st.sidebar.selectbox('Choose Your Model', config.classifier_models)
    
    # Load Model & Metrics Summary
    classifier = joblib.load(f'./model/{usr_model}.pkl')
    metrics_df = pd.read_csv('./model/results.csv')
    
    # Print the Header Banner
    html_str = f"""
                    <h3 style="background-color:#00A1DE; text-align:center; font-family:arial;color:white">MODEL METRICS: {classifier.__class__.__name__} </h3>
                """

    st.markdown(html_str, unsafe_allow_html=True)
    st.divider()
    
    # Show Metrics
    st.dataframe(metrics_df[metrics_df.Classifier == usr_model][['Classifier', 'Accuracy', 'Precision',
       'Recall', 'F1_Score', 'ROC_AUC_Scr', 'CV_Score']], hide_index=True, use_container_width=True)
    st.divider()
    
    # Show Graphs
    col1, col2= st.columns(2)
    with col1:
        # plt_conf_mtx = plot_confusion_matrix(classifier, X_test, y_test, display_labels=['No Fraud', 'Fraud'], cmap='YlGnBu')
        # plt_conf_mtx.ax_.set_title("Confusion Matrix")
        st.write("Confusion Matrix")
        predictions = classifier.predict(X_test)
        cm = confusion_matrix(y_test, predictions, labels=classifier.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                    display_labels=['No Fraud', 'Fraud'])
        disp.plot()
        st.pyplot()

    with col2:
        st.write("Area Under Curve(AUC) - TPR vs FPR")
        disp = RocCurveDisplay.from_estimator(classifier, X_test, y_test)
        disp.plot()
        st.pyplot()

    st.divider()
    col3, col4= st.columns(2)
    with col3:
        st.write("Precision Recall Curve")
        disp = PrecisionRecallDisplay.from_estimator(classifier, X_test, y_test)
        disp.plot()
        st.pyplot()
    
    with col4:
        
        if usr_model == ( 'VotingClassifier'):
            st.write('Permutation Importance')
            perm = PermutationImportance(classifier, random_state=1).fit(X_train, y_train)
            html_object  = eli5.show_weights(perm, feature_names = X_train.columns.tolist())
            components.v1.html(html_object._repr_html_(), width=500, height=300, scrolling=False)
        
        elif usr_model == ( 'StackingClassifier'):
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
    
    # Load Input Data
    input_df = pd.read_csv("./data/Base.csv")
    input_df_num = map_categorical_column(input_df)

    X_train, y_train, X_test, y_test = create_sample_set(input_df_num, 1)
    
    streamlit_interface(X_train, y_train, X_test, y_test)

    
    