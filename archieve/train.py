## Code for Modelling
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import accuracy_score, confusion_matrix # Library for model evaluation
from sklearn import metrics
from sklearn.model_selection import train_test_split # Library to split datset into test and train
from sklearn.ensemble  import RandomForestClassifier # Random Forest Classifier
from sklearn.ensemble import AdaBoostClassifier # Ada Boost Classifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model  import LogisticRegression # Logistic Regression Classifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib

def create_sample_set(train_df, non_fraud_sample_sizse):
    
    ''' Function to Create Data for Modelling '''
    
    # Select columns
    train_df = train_df[['prev_address_months_count', 'date_of_birth_distinct_emails_4w','credit_risk_score', 'bank_months_count', 
                         'proposed_credit_limit','customer_age', 'housing_status','device_os', 'employment_status',
                         'keep_alive_session','has_other_cards','phone_home_valid','payment_type', 'fraud_bool', 'month']]
    
    # Fraud Transactions
    train_df_fraud = train_df[train_df.fraud_bool == 1]
    print(f'Shape of train_df_fraud {train_df_fraud.shape}')
    
    # Non Fraud Transactions
    train_df_non_fraud = train_df[train_df.fraud_bool == 0].sample(train_df_fraud.shape[0] * non_fraud_sample_sizse)
    print(f'Shape of train_df_non_fraud {train_df_non_fraud.shape}')
    
    # Merge Fraud & Non Fraud
    train_df_merged = pd.concat([train_df_fraud, train_df_non_fraud])

    # Shuffle
    train_df_merged.iloc[:,:] = train_df_merged.sample(frac=1,random_state=123,ignore_index=True)
    print(f'After merge & shuffle Shape of train_df_merged {train_df_merged.shape}')
#     print(f'Value Counts:\n {train_df_merged["fraud_bool"].value_counts(normalize=True)}')
    
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

# Function for Precsion, Recall and F1 Score
def calc_classfier_metric(classifier, y_test, y_pred):
    '''
    Function for Precsion, Recall and F1 Score
    '''
    accuracy      = accuracy_score(y_test, y_pred)
    precision     = precision_score(y_test, y_pred)
    recall        = recall_score(y_test, y_pred)
    F1_score      = f1_score(y_test, y_pred)
    roc_auc_scr   = roc_auc_score(y_test, y_pred)
    conf_mat      = confusion_matrix(y_test, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    
    return accuracy, precision, recall, F1_score, roc_auc_scr, conf_mat, fpr, tpr

# Build Classification Model
def build_individual_classifier_model(X_train, X_test, y_train, y_test, classifier_model):
    '''
    Function to Build Classification Model for Individual Classifier
    '''
    print('Into build_individual_classifier_model')
    
    classifier_performance = []
    cnf_lst = []

    for classifier in classifier_model:

        # Fitting the training set into classification model
        classifier.fit(X_train,y_train)

        # Predicting the output on test datset
        y_pred = classifier.predict(X_test)    

        # Cross Validation Score on training test
        cv = RepeatedStratifiedKFold(n_splits=5, random_state=42)
        scores = cross_val_score(classifier, X_train,y_train, cv=5, scoring='f1_weighted')
        cv_score_mean = scores.mean()

        # Classification score
        accuracy, precision, recall, F1_score, roc_auc_scr, conf_mat, fpr, tpr = calc_classfier_metric(classifier, y_test, y_pred)
        classifier_performance.append([classifier.__class__.__name__, conf_mat, accuracy, precision, recall, F1_score, roc_auc_scr, cv_score_mean, fpr, tpr])
        
        # Store the model into pkl
        joblib.dump(classifier, f'./model/{classifier.__class__.__name__}.pkl')
     
    class_perf_df = pd.DataFrame(classifier_performance, columns=['Classifier', 'Conf_Mtrx', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC_Scr', 'CV_Score', 'FPR', 'TPR']).sort_values('F1_Score', ascending = False)
    
    return class_perf_df

def build_voting_classifier_model(X_train, X_test, y_train, y_test, classifier_model, ind_class_model_df):
    
    '''
    Function to Classifier Model for Voting Classifier
    '''
    
    print('Into build_voting_classifier_model')
    
    classifier_performance = []
    cnf_lst = []

    # Voting Classifier                
    clf1 = classifier_model[1]
    clf2 = classifier_model[2]
    clf3 = classifier_model[3]
    
    vote_classifier = VotingClassifier(
                                        estimators=[('ada', clf1),('xgb', clf2), ('lgb', clf3)],
                                        voting='soft'
                                    )
    
    # Fitting the training set into classification model
    vote_classifier.fit(X_train,y_train)

    # Predicting the output on test datset
    y_pred = vote_classifier.predict(X_test)    

    # Cross Validation Score on training test
    cv = RepeatedStratifiedKFold(n_splits=5, random_state=42)
    scores = cross_val_score(vote_classifier, X_train,y_train, cv=5, scoring='f1_weighted')
    cv_score_mean = scores.mean()

    # Classification score
    accuracy, precision, recall, F1_score, roc_auc_scr, conf_mat, fpr, tpr = calc_classfier_metric(vote_classifier, y_test, y_pred)
    classifier_performance.append([vote_classifier.__class__.__name__, conf_mat, accuracy, precision, recall, F1_score, roc_auc_scr, cv_score_mean, fpr, tpr])
    
    # Store the model into pkl
    joblib.dump(vote_classifier, f'./model/{vote_classifier.__class__.__name__}.pkl')
        
    class_perf_df = pd.DataFrame(classifier_performance, columns=['Classifier', 'Conf_Mtrx', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC_Scr', 'CV_Score', 'FPR', 'TPR']).sort_values('F1_Score', ascending = False)
    
    voting_class_df = pd.concat([ind_class_model_df, class_perf_df])
    
    return voting_class_df

# Build Classification Model
def build_stacking_classifier_model(X_train, X_test, y_train, y_test, classifier_model, prev_class_model_df):
    
    '''
    Function to Classifier Model for Voting Classifier
    '''
    
    print('Into build_stacking_classifier_model')
    
    classifier_performance = []
    cnf_lst = []

    # Voting Classifier                
    clf1 = classifier_model[1]
    clf2 = classifier_model[2]
    clf3 = classifier_model[3]
    
    stacking_classifier = StackingClassifier(
                                                estimators = [('ada', clf1),('xgb', clf2), ('lgb', clf3)],
                                                final_estimator = LogisticRegression(),
                                                cv = 5
                                    )
    
    
    # Fitting the training set into classification model
    stacking_classifier.fit(X_train,y_train)

    # Predicting the output on test datset
    y_pred = stacking_classifier.predict(X_test)    

    # Cross Validation Score on training test
    cv = RepeatedStratifiedKFold(n_splits=5, random_state=42)
    scores = cross_val_score(stacking_classifier, X_train,y_train, cv=5, scoring='f1_weighted')
    cv_score_mean = scores.mean()

    # Classification score
    accuracy, precision, recall, F1_score, roc_auc_scr, conf_mat, fpr, tpr = calc_classfier_metric(stacking_classifier, y_test, y_pred)
    classifier_performance.append([stacking_classifier.__class__.__name__, conf_mat, accuracy, precision, recall, F1_score, roc_auc_scr, cv_score_mean, fpr, tpr])
    
    # Store the model into pkl
    joblib.dump(stacking_classifier, f'./model/{stacking_classifier.__class__.__name__}.pkl')        
    class_perf_df = pd.DataFrame(classifier_performance, columns=['Classifier', 'Conf_Mtrx', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC_Scr', 'CV_Score', 'FPR', 'TPR']).sort_values('F1_Score', ascending = False)
    
    stacking_class_df = pd.concat([prev_class_model_df, class_perf_df])
    
    return stacking_class_df

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

def perform_model_training():
    
    input_df = pd.read_csv("./data/Base.csv")
    print(input_df.shape)

    input_df_copy = input_df.copy()
    input_df_num = map_categorical_column(input_df_copy)

    X_train, y_train, X_test, y_test = create_sample_set(input_df_num, 1)
    print('There are {} samples in the training set and {} samples in the test set'.format(X_train.shape[0], X_test.shape[0]))

    # Machine Learning Model Build
    classifier_model = [
                        RandomForestClassifier(random_state=42), 
                        AdaBoostClassifier(learning_rate = 0.1, n_estimators=500, random_state=42), 
                        XGBClassifier(colsample_bytree=1.0, gamma=5, learning_rate=1.0, max_depth=5, min_child_weight=1,    n_estimators=10, subsample=1.0, random_state=42),
                        LGBMClassifier(boosting_type = 'dart', colsample_bytree=1.0, learning_rate = 0.1, max_depth=10,n_estimators = 50, subsample=0.6, num_leaves = (2^5-1), random_state=42)
                    ]

    # Call Classification module
    ind_class_model_df        = build_individual_classifier_model(X_train, X_test, y_train, y_test, classifier_model)
    ind_voting_model_df       = build_voting_classifier_model(X_train, X_test, y_train, y_test, classifier_model,   ind_class_model_df)
    ind_voting_stack_model_df = build_stacking_classifier_model(X_train, X_test, y_train, y_test, classifier_model, ind_voting_model_df)

    # Model_Evaluation
    ind_voting_stack_model_df.to_csv('./model/results.csv')
    
    print('End of Code')
    
if __name__ == '__main__':
    perform_model_training()