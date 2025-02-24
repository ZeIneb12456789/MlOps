import argparse
import os
import pickle
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models.signature import infer_signature
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
from sklearn.metrics import classification_report

# Default file paths and target column
DEFAULT_TRAIN_FILE = "/mnt/c/Users/MSI/Desktop/churn-bigml-80.csv"
DEFAULT_TEST_FILE = "/mnt/c/Users/MSI/Desktop/churn-bigml-20.csv"
DEFAULT_TARGET = "Churn"
DEFAULT_MODEL_FILE = "trained_model.joblib"
DEFAULT_ENCODER_FILE = "encoders.pkl"
DEFAULT_DATA_FILE = "prepared_data.pkl"  # File for saving preprocessed data

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
    parser.add_argument(
        "--data_file",
        type=str,
        default=DEFAULT_DATA_FILE,
        help="File path to save/load prepared data",
    )

    args = parser.parse_args()

    mlflow.set_experiment("Churn Prediction")  # Set MLflow experiment name

    if args.command == "prepare":
        # Process data once and save it
        X_train, X_test, y_train, y_test, label_encoders = prepare_data(
            args.train_file, args.test_file, args.target
        )
        
        # Save processed data to avoid re-processing
        with open(args.data_file, "wb") as f:
            pickle.dump((X_train, X_test, y_train, y_test), f)

        save_encoders(label_encoders, args.encoder_file)
        print("✅ Data preparation complete. Encoders and processed data saved.")

    elif args.command == "train":
        # Load preprocessed data
        if not os.path.exists(args.data_file):
            print("❌ Error: Prepared data not found. Run 'prepare' first.")
            return
        
        with open(args.data_file, "rb") as f:
            X_train, X_test, y_train, y_test = pickle.load(f)

        with mlflow.start_run() as run:  # Ensures run is active
            print(f"✅ MLflow Run ID: {run.info.run_id}")  # Debugging step
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)

            # ✅ Explicitly log parameters BEFORE training
            print("🔹 Logging parameters to MLflow...")
            mlflow.log_param("n_estimators", model.n_estimators)
            mlflow.log_param("random_state", model.random_state)
            print("✅ Parameters logged successfully.")

            # Train the model
            model = train_model(model, X_train, y_train)

            save_model(model, args.model_file)

            # Convert input example to float to avoid MLflow schema warning
            input_example = pd.DataFrame(X_train[:1]).astype(float)

            # Infer model signature with data converted to float
            signature = infer_signature(X_train.astype(float), model.predict(X_train))

            # ✅ Log the model with MLflow
            mlflow.sklearn.log_model(
                model,
                "random_forest_model",
                input_example=input_example,
                signature=signature
            )

            print("✅ Model training complete. Model saved.")

    elif args.command == "evaluate":
        if not os.path.exists(args.model_file):
            print("❌ Error: Model not found. Train the model first.")
            return

        if not os.path.exists(args.data_file):
            print("❌ Error: Prepared data not found. Run 'prepare' first.")
            return

        with open(args.data_file, "rb") as f:
            X_train, X_test, y_train, y_test = pickle.load(f)

        with mlflow.start_run() as run:
            print(f"✅ MLflow Run ID for evaluation: {run.info.run_id}")

            model = load_model(args.model_file)

            # Generate predictions and classification report
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)

            # ✅ Log relevant numerical metrics in MLflow
            print("🔹 Logging evaluation metrics to MLflow...")
            mlflow.log_metric("accuracy", report["accuracy"])
            mlflow.log_metric("precision_True", report["True"]["precision"])
            mlflow.log_metric("recall_True", report["True"]["recall"])
            mlflow.log_metric("f1_True", report["True"]["f1-score"])
            mlflow.log_metric("precision_False", report["False"]["precision"])
            mlflow.log_metric("recall_False", report["False"]["recall"])
            mlflow.log_metric("f1_False", report["False"]["f1-score"])
            print("✅ Metrics logged successfully.")

            print("✅ Evaluation complete. Metrics logged in MLflow.")

if __name__ == "__main__":
    main()

