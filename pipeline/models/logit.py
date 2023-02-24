import statsmodels.api as sm

def apply_ols(training_data, test_data, config):

    Y_train = training_data.loan_status
    X_train = training_data.drop(['loan_status'], axis=1)
    X_train = sm.add_constant(X_train)
    log_regression = sm.Logit(Y_train,X_train)
    results = log_regression.fit()
    results.summary()

    return