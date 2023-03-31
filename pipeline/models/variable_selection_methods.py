
def apply_selection_lasso():
    return 

def apply_selection_pca(data):
    return 0


def apply_variable_selection(training_data, test_data, selection_method):

    variable_selections_methods = {
        "lasso": apply_selection_lasso,
        "pca": apply_selection_pca
    }

    selection = variable_selections_methods[selection_method](training_data)

    #filter training and test data with selection

    return #training and test data