import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import mlflow
import mlflow.sklearn

def train_and_evaluate_model(train_filepath, test_filepath):
    # Load training and testing datasets
    train_df = pd.read_csv(train_filepath)
    test_df = pd.read_csv(test_filepath)

    # Separate features and target variable ('Class')
    X_train = train_df.drop('Class', axis=1)
    y_train = train_df['Class']
    X_test = test_df.drop('Class', axis=1)
    y_test = test_df['Class']

    # Convert integer columns to float64
    X_train = X_train.astype({col: 'float64' for col in X_train.select_dtypes(include='int').columns})

    # Start an MLflow run for experiment tracking
    with mlflow.start_run():
        # Create and train a logistic regression model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Make predictions on the test data
        predictions = model.predict(X_test)
        
        # Compute evaluation metrics
        report = classification_report(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        # Print metrics
        print("Classification Report:")
        print(report)
        print("Confusion Matrix:")
        print(cm)
        print("ROC AUC Score:")
        print(roc_auc)

        # Create an input example using the first row of the training data
        input_example = X_train.iloc[:1]

        # Log parameters, metrics, and the model with MLflow
        mlflow.log_param("max_iter", 1000)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.sklearn.log_model(model, "model", input_example=input_example)

if __name__ == "__main__":
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Ensure the data directory exists
    data_dir = os.path.join(script_dir, '../data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Filepaths for the preprocessed training and testing data
    train_filepath = os.path.join(data_dir, 'train_preprocessed.csv')
    test_filepath = os.path.join(data_dir, 'test_preprocessed.csv')
    
    # Train the model and evaluate it
    train_and_evaluate_model(train_filepath, test_filepath)
