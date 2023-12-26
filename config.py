payment_type        = ('AA', 'AB', 'AC', 'AD', 'AE')
employment_status   = ('CA', 'CB', 'CC', 'CD', 'CE','CF','CG')
housing_status      = ('BA', 'BB', 'BC', 'BD', 'BE','BF','BG')
device_os           = ('windows','other','linux','macintosh','x11')
customer_age        = ('< 50 yrs', '>= 50 yrs')

classifier_models   = ('LGBMClassifier', 
                       'XGBClassifier', 
                       'AdaBoostClassifier', 
                       'VotingClassifier', 
                       'StackingClassifier')

remove_nodes = ('INTERNET', 'TELEAPP', 'linux', 'macintosh', 'other','windows', 'x11', 'AA', 'AB', 'AC', 'AD', 'AE')
file_list = ['Base.csv', 'variant_1.csv', 'variant_2.csv', 'variant_3.csv', 'variant_4.csv', 'variant_5.csv']
sampling_strategy = ['1_1', '1_2', '1_3']
sampling_strategy_dict = {'1_1':1, '1_2':2, '1_3':3}

reqd_col_modelling = ['housing_status',
                            'device_os',
                            'credit_risk_score',
                            'current_address_months_count',
                            'has_other_cards',
                            'keep_alive_session',
                            'prev_address_months_count',
                            'phone_home_valid',
                            'proposed_credit_limit',
                            'name_email_similarity',
                            'income',
                            'fraud_bool' 
                        ]
