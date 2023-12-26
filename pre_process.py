import pandas as pd
import numpy as np

def map_and_fmt_categorical_column(model_columns):
    
    ''' Function to map the categorical columns '''
    
    map_housing_status    = {'BA':0, 'BB':1, 'BC':2, 'BD':3, 'BE':4,'BF':5,'BG':6}
    map_device_os         = {'windows':0,'other':1,'linux':2,'macintosh':3,'x11':4} 

    num_housing_status      = map_housing_status[model_columns[0]]
    num_device_os           = map_device_os[model_columns[1]]

    num_proposed_credit_limit      = model_columns[2]
    num_income                     = model_columns[3]

    num_has_other_cards         = int(model_columns[4])
    num_keep_alive_session      = int(model_columns[5])
    num_phone_home_valid        = int(model_columns[6])

    num_name_email_similarity           = float(model_columns[7])
    num_current_address_months_count    = int(model_columns[8])
    num_prev_address_months_count       = int(model_columns[9])
    num_credit_risk_score               = int(model_columns[10])  

    inference_layout = [
                                num_housing_status,
                                num_device_os,
                                num_credit_risk_score,
                                num_current_address_months_count,
                                num_has_other_cards,
                                num_keep_alive_session,
                                num_prev_address_months_count,
                                num_phone_home_valid,
                                num_proposed_credit_limit,
                                num_name_email_similarity,
                                num_income
                            ]
    print(                                num_housing_status,
                                num_device_os,
                                num_credit_risk_score,
                                num_current_address_months_count,
                                num_has_other_cards,
                                num_keep_alive_session,
                                num_prev_address_months_count,
                                num_phone_home_valid,
                                num_proposed_credit_limit,
                                num_name_email_similarity,
                                num_income)
    return np.array(inference_layout).reshape(1, -1)

