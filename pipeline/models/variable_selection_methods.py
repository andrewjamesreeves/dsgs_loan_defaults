
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

def apply_selection_lasso(training_data):

    Y_train = training_data[['loan_status']]
    X_train = training_data.drop(['loan_status'], axis=1)

    lasso = LassoCV()

    lasso_fit = lasso.fit(X_train, Y_train)
    lasso_coefs = dict(zip(list(X_train.columns), list(lasso_fit.coef_)))

    selected_indicators = list({k:v for k,v in lasso_coefs.items() if v != float(0)}.keys())

    return selected_indicators

def apply_random_forest(training_data):

    Y_train = training_data[['loan_status']]
    X_train = training_data.drop(['loan_status'], axis=1)

    forest_model = SelectFromModel(RandomForestRegressor(n_estimators=100))
    forest_model.fit(X_train, Y_train)
    selected_indicators = X_train.columns[(forest_model.get_support())]

    return list(selected_indicators)

def apply_selection_pca(data):
    return 0


def apply_variable_selection(training_data, test_data, selection_method):

    variable_selections_methods = {
        "lasso": apply_selection_lasso,
        "pca": apply_selection_pca,
        "rf": apply_random_forest
    }

    selection = variable_selections_methods[selection_method](training_data)

    #append_outcome_variable
    selection.append('loan_status')
    #filter training and test data with selection
    training_data_filtered = training_data[selection]
    test_data_filtered = test_data[selection]


    return training_data_filtered, test_data_filtered