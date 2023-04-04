from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression



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
    sel_.fit(X_train, Y_train)
    
    # obtain list of features with coefficient > 0 
    remove_feats = list(X_train.columns[(sel_.estimator_.coef_ == 0).ravel().tolist()])
    
    return remove_feats

def apply_selection_pca(data):
    return 0


def apply_variable_selection(training_data, test_data, selection_method):

    variable_selections_methods = {
        "lasso": apply_selection_lasso,
        "pca": apply_selection_pca
    }

    # obtain list of features to remove
    selection_remove = variable_selections_methods[selection_method](training_data)

    #filter training and test data 
    training_data_selected = training_data.drop(columns=selection_remove)
    test_data_selected = test_data.drop(columns=selection_remove)

    return training_data_selected, training_data_selected