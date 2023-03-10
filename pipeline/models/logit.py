import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report




def apply_logit(training_data, test_data, config):

    # Split data
    Y_train = training_data[['loan_status']]
    X_train = training_data.drop(['loan_status'], axis=1)

    Y_test = test_data[['loan_status']]
    X_test = test_data.drop(['loan_status'], axis=1)
    
    # Define parameters  
    logistic_regression_model = LogisticRegression(solver=config['solver'],  
                                        C=config['C'], 
                                        random_state=config['random_state'], 
                                        class_weight=config['class_weight'])
            
    # Fit the model
    logistic_regression_model.fit(X_train, Y_train)

    # Predict values using the test data.
    yhat = logistic_regression_model.predict(X_test)
    
    # Create dataframe with predicted and actual values    
    yhat_df = pd.DataFrame(yhat, columns = ['y_pred']).sort_index()
    
    error_df = Y_test.merge(yhat_df, left_index=True, right_index=True)
    error_df['correct'] = error_df.y_pred == error_df.loan_status

    # Produce evaluation metrics
    target_names = ["No default", "Default"] # Set the names for our report to produce.

    # Generate the report using the target test and prediction values.
    classif_report = classification_report(Y_test, yhat, target_names=target_names)
    
    print(classif_report)
    
    return 


