from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd

def apply_rfc(training_data, test_data, config):
    
    # Split data
    Y_train = training_data[['loan_status']]
    X_train = training_data.drop(['loan_status'], axis=1)

    Y_test = test_data[['loan_status']]
    X_test = test_data.drop(['loan_status'], axis=1)
    
    # Fit the model
    rfc = RandomForestClassifier()
    rfc.fit(X_train, Y_train)
    
    # Predict values using the test data.
    yhat = rfc.predict(X_test)
    
    # Generate the report using the target test and prediction values.
    classif_report = classification_report(Y_test, yhat, target_names=["No default", "Default"])
    
    print(pd.DataFrame(classif_report))
    
    return
    
    