import statsmodels.api as sm
import pandas as pd

def apply_logit(training_data, test_data, config):

    Y_train = training_data[['loan_status']]
    X_train = training_data.drop(['loan_status'], axis=1).iloc[:,0:33]
    X_train = sm.add_constant(X_train)

    Y_test = test_data[['loan_status']]
    X_test = test_data.drop(['loan_status'], axis=1).iloc[:,0:33]
    X_test = sm.add_constant(X_test)

    
    log_regression = sm.Logit(Y_train,X_train)
    results = log_regression.fit()

    yhat = round(results.predict(X_test)).to_frame(name='y_pred').sort_index()

    error_df = Y_test.merge(yhat, left_index=True, right_index=True)
    error_df['correct'] = error_df.y_pred == error_df.loan_status

    error = len(error_df[error_df.correct == True])/len(error_df)

    return