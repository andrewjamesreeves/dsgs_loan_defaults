import pandas as pd
import numpy as np
from models.variable_selection_methods import apply_variable_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.metrics import classification_report



def split_data(training_data, test_data):
    # Split data
    Y_train = training_data[['loan_status']]
    X_train = training_data.drop(['loan_status'], axis=1)

    Y_test = test_data[['loan_status']]
    X_test = test_data.drop(['loan_status'], axis=1)
    
    return X_train, X_test, Y_train, Y_test


def fit_and_predict(chosen_model, X_train, Y_train, X_test):
    # Fit the model
    chosen_model.fit(X_train, Y_train)

    # Predict values using the test data.
    yhat = chosen_model.predict(X_test)
    
    return yhat, chosen_model


    
def pred_actual_df(yhat, Y_test):
    # Create dataframe with predicted and actual values    
    yhat_df = pd.DataFrame(yhat, columns = ['y_pred']).sort_index()
    
    error_df = Y_test.merge(yhat_df, left_index=True, right_index=True)
    error_df['correct'] = error_df.y_pred == error_df.loan_status
    
    return error_df


def apply_logit(training_data, test_data, config):

    if config['variable_selection'] != str('False'):
        training_data, test_data = apply_variable_selection(training_data, test_data, config['variable_selection'])

    # Split data
    X_train, X_test, Y_train, Y_test = split_data(training_data, test_data)
    
    # Define parameters  
    logistic_regression_model = LogisticRegression(solver=config['solver'],  
                                        C=config['C'], 
                                        random_state=config['random_state'], 
                                        class_weight=config['class_weight'])
            
    yhat, chosen_model = fit_and_predict(logistic_regression_model, X_train, Y_train, X_test)
   
    # Create dataframe with predicted and actual values    
    error_df = pred_actual_df(yhat, Y_test)
   
     # Generate the report using the target test and prediction values.
    classif_report = classification_report(Y_test, yhat, target_names=["No default", "Default"])
    print(classif_report)
   
    return Y_test, yhat, chosen_model



def apply_rfc(training_data, test_data, config):
    
    if config['variable_selection'] != str('False'):
        training_data, test_data = apply_variable_selection(training_data, test_data, config['variable_selection'])
    
    # Split data
    X_train, X_test, Y_train, Y_test = split_data(training_data, test_data)
    
    # Fit the model
    rfc = RandomForestClassifier()
    
    yhat, chosen_model = fit_and_predict(rfc, X_train, Y_train, X_test)
    
    # Create dataframe with predicted and actual values    
    error_df = pred_actual_df(yhat, Y_test)
    
    # Generate the report using the target test and prediction values.
    classif_report = classification_report(Y_test, yhat, target_names=["No default", "Default"])
    print(classif_report)
    
    return Y_test, yhat, chosen_model



def apply_lasso(training_data, test_data, config):

    # Split data
    X_train, X_test, Y_train, Y_test = split_data(training_data, test_data)
    
    # Fit the model
    logistic_lasso_model = LogisticRegression(penalty = config['penalty'],
                                    solver=config['solver'],  
                                    C=config['C'], 
                                    random_state=config['random_state'], 
                                    class_weight=config['class_weight']
                                    )
    
    yhat, chosen_model = fit_and_predict(logistic_lasso_model, X_train, Y_train, X_test)
    
    # Create dataframe with predicted and actual values    
    error_df = pred_actual_df(yhat, Y_test)
    
    # Generate the report using the target test and prediction values.
    classif_report = classification_report(Y_test, yhat, target_names=["No default", "Default"])
    print(classif_report)
    
    return Y_test, yhat, chosen_model
    
