import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib


def prepare_data(train_file_path, test_file_path, target):
    """
    Prepare the data by loading, cleaning, and encoding it.
    """
    print("Loading training and test data...")
    train_data = pd.read_csv(train_file_path)
    test_data = pd.read_csv(test_file_path)
    print("Data loaded successfully.")

    # Drop missing values
    print("Dropping missing values...")
    train_data = train_data.dropna()
    test_data = test_data.dropna()
    print("Missing values dropped.")

    # Encode categorical columns (including the target if it's categorical)
    print("Encoding categorical columns...")
    label_encoders = {}
    for column in train_data.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        train_data[column] = le.fit_transform(train_data[column])

        # Handle unseen categories in test data by encoding them as -1
        test_data[column] = test_data[column].apply(
            lambda x: x if x in le.classes_ else "unseen"
        )
        test_data[column] = le.transform(test_data[column])

        label_encoders[column] = le
    print("Categorical columns encoded.")

    # Split features and target
    print("Splitting features and target...")
    X_train = train_data.drop(target, axis=1)
    y_train = train_data[target]
    X_test = test_data.drop(target, axis=1)
    y_test = test_data[target]
    print("Data preparation complete.")

    return X_train, X_test, y_train, y_test, label_encoders


def train_model(model, X_train, y_train):
    """
    Train the model on the training data.
    """
    print("Training the model...")
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and print a classification report.
    """
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print("Model evaluation complete.")
    print("Classification Report:")
    print(report)
    return report


def save_model(model, file_path):
    """
    Save the trained model to a file.
    """
    print(f"Saving the model to {file_path}...")
    joblib.dump(model, file_path)
    print("Model saved successfully.")


def save_encoders(label_encoders, file_path):
    """
    Save the label encoders to a file.
    """
    print(f"Saving label encoders to {file_path}...")
    joblib.dump(label_encoders, file_path)
    print("Label encoders saved successfully.")


def load_model(file_path):
    """
    Load a trained model from a file.
    """
    print(f"Loading the model from {file_path}...")
    model = joblib.load(file_path)
    print("Model loaded successfully.")
    return model


def load_encoders(file_path):
    """
    Load label encoders from a file.
    """
    print(f"Loading label encoders from {file_path}...")
    label_encoders = joblib.load(file_path)
    print("Label encoders loaded successfully.")
    return label_encoders
