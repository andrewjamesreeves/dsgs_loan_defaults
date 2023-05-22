from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

def apply_selection_lasso(training_data):
    
    # Split data
    Y_train = training_data[['loan_status']]
    X_train = training_data.drop(['loan_status'], axis=1)
    
    
    # Fit the model
    logistic_lasso_model = LogisticRegression(penalty = "l1",
                                    solver="liblinear",  
                                    C=0.001, 
                                    random_state=1234, 
                                    class_weight="balanced"
                                    )
    
    # fit lasso model using SelectfromModel
    sel_ = SelectFromModel(logistic_lasso_model)
    sel_.fit(X_train, Y_train.values.ravel())
    
    # obtain list of features with coefficient > 0 
    selected_indicators = list(X_train.columns[(sel_.estimator_.coef_ != 0).ravel().tolist()])
    
    return selected_indicators

def apply_random_forest(training_data):

    Y_train = training_data[['loan_status']]
    X_train = training_data.drop(['loan_status'], axis=1)

    forest_model = SelectFromModel(RandomForestRegressor(n_estimators=100))
    forest_model.fit(X_train, Y_train.values.ravel())
    selected_indicators = X_train.columns[(forest_model.get_support())]

    return list(selected_indicators)

def apply_selection_pca(data):
    return 0


def apply_variable_selection(training_data, test_data, selection_method):

    variable_selections_methods = {
        "lasso": apply_selection_lasso,
        "pca": apply_selection_pca,
        "random_forest": apply_random_forest
    }

    # obtain list of features to remove
    selection = variable_selections_methods[selection_method](training_data)

    #append_outcome_variable
    selection.append('loan_status')
    #filter training and test data with selection
    training_data_filtered = training_data[selection]
    test_data_filtered = test_data[selection]

    return training_data_filtered, test_data_filtered