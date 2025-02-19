import argparse
import os
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    save_encoders,
    load_model,
    load_encoders,
)
from sklearn.ensemble import RandomForestClassifier

# Default file paths and target column
DEFAULT_TRAIN_FILE = "/mnt/c/Users/MSI/Desktop/churn-bigml-80.csv"
DEFAULT_TEST_FILE = "/mnt/c/Users/MSI/Desktop/churn-bigml-20.csv"
DEFAULT_TARGET = "Churn"
DEFAULT_MODEL_FILE = "trained_model.joblib"
DEFAULT_ENCODER_FILE = "encoders.pkl"


def main():
    parser = argparse.ArgumentParser(description="Machine Learning Model Pipeline")
    parser.add_argument(
        "command", choices=["prepare", "train", "evaluate"], help="Action to perform"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=DEFAULT_TRAIN_FILE,
        help="Path to the training data file",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=DEFAULT_TEST_FILE,
        help="Path to the test data file",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=DEFAULT_TARGET,
        help="Target column for classification",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default=DEFAULT_MODEL_FILE,
        help="File path to save/load the trained model",
    )
    parser.add_argument(
        "--encoder_file",
        type=str,
        default=DEFAULT_ENCODER_FILE,
        help="File path to save/load encoders",
    )

    args = parser.parse_args()

    if args.command == "prepare":
        X_train, X_test, y_train, y_test, label_encoders = prepare_data(
            args.train_file, args.test_file, args.target
        )
        save_encoders(label_encoders, args.encoder_file)
        print("Data preparation complete. Encoders saved.")

    elif args.command == "train":
        X_train, X_test, y_train, y_test, label_encoders = prepare_data(
            args.train_file, args.test_file, args.target
        )
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model = train_model(model, X_train, y_train)
        save_model(model, args.model_file)
        save_encoders(label_encoders, args.encoder_file)
        print("Model training complete. Model and encoders saved.")

    elif args.command == "evaluate":
        if not os.path.exists(args.model_file) or not os.path.exists(args.encoder_file):
            print("Error: Model or encoders not found. Train the model first.")
            return

        model = load_model(args.model_file)
        label_encoders = load_encoders(args.encoder_file)
        X_train, X_test, y_train, y_test, _ = prepare_data(
            args.train_file, args.test_file, args.target
        )
        evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
