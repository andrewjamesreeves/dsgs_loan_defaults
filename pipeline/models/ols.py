import statsmodels.api as sm

def apply_ols(training_data, test_data, config):

    Y = training_data.loan_status
    X = training_data.drop(['loan_status'], axis=1)
    X = sm.add_constant(X)
    model = sm.OLS(Y,X)
    results = model.fit()
    results.summary()

    return