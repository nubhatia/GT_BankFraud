import numpy as np 
import pandas as pd
import joblib

def calc_feature_importance(model_name):
    
    train_columns = ['prev_address_months_count',
                    'date_of_birth_distinct_emails_4w',
                    'credit_risk_score',
                    'bank_months_count',
                    'proposed_credit_limit',
                    'customer_age',
                    'housing_status',
                    'device_os',
                    'employment_status',
                    'keep_alive_session',
                    'has_other_cards',
                    'phone_home_valid',
                    'payment_type'
                    ]
    try:
        loaded_model  = joblib.load(f'./model/{model_name}.pkl')
        feature_importances = pd.DataFrame(
                                            loaded_model.feature_importances_,
                                            index = train_columns,
                                            columns=['Feature_Importance']).sort_values('Feature_Importance', ascending=True
                                        )
                                            
        return feature_importances.reset_index().rename(columns={'index':'Feature'})
    
    except Exception as e:
        print('Unble to load model:', e)
        return e
    
# if __name__ == '__main__':
#     model_name = 'XGBClassifier'
#     calc_feature_importance(model_name)