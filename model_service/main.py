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
import psutil
import time

# Default file paths and target column
DEFAULT_TRAIN_FILE = "/mnt/c/Users/MSI/Desktop/churn-bigml-80.csv"
DEFAULT_TEST_FILE = "/mnt/c/Users/MSI/Desktop/churn-bigml-20.csv"
DEFAULT_TARGET = "Churn"
DEFAULT_MODEL_FILE = "trained_model.joblib"
DEFAULT_ENCODER_FILE = "encoders.pkl"
DEFAULT_DATA_FILE = "prepared_data.pkl"  # File for saving preprocessed data
mlflow.set_tracking_uri("http://localhost:5000")

def log_system_metrics(start_time, initial_cpu, initial_memory):
    """Helper function to log system metrics to MLflow."""
    final_cpu = psutil.cpu_percent(interval=1)
    final_memory = psutil.virtual_memory().percent

    # Log system metrics to MLflow
    mlflow.log_metric("initial_cpu_usage", initial_cpu)
    mlflow.log_metric("initial_memory_usage", initial_memory)
    mlflow.log_metric("final_cpu_usage", final_cpu)
    mlflow.log_metric("final_memory_usage", final_memory)

    # Log training time
    training_time = time.time() - start_time
    mlflow.log_metric("training_time_seconds", training_time)
    
    print(f"Training Time: {training_time} seconds")
    print(f"CPU Usage During Training: {initial_cpu} -> {final_cpu}")
    print(f"Memory Usage During Training: {initial_memory} -> {final_memory}")

    return training_time, final_cpu, final_memory


def log_inference_metrics(model, X_test):
    """Helper function to log inference time and system metrics."""
    # Log inference time
    inference_start = time.time()
    model.predict(X_test)
    inference_time = time.time() - inference_start
    mlflow.log_metric("inference_time_seconds", inference_time)

    # Log system metrics after inference
    final_cpu_inference = psutil.cpu_percent(interval=1)
    final_memory_inference = psutil.virtual_memory().percent

    mlflow.log_metric("final_cpu_usage_inference", final_cpu_inference)
    mlflow.log_metric("final_memory_usage_inference", final_memory_inference)

    print(f"Inference Time: {inference_time} seconds")
    print(f"CPU Usage After Inference: {final_cpu_inference}")
    print(f"Memory Usage After Inference: {final_memory_inference}")


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

            # Log parameters
            print("🔹 Logging parameters to MLflow...")
            mlflow.log_param("n_estimators", model.n_estimators)
            mlflow.log_param("random_state", model.random_state)
            print("✅ Parameters logged successfully.")

            # Log initial system metrics before training
            initial_cpu = psutil.cpu_percent(interval=1)
            initial_memory = psutil.virtual_memory().percent
            start_time = time.time()

            # Train the model
            model = train_model(model, X_train, y_train)

            # Log system metrics and training time
            training_time, final_cpu, final_memory = log_system_metrics(start_time, initial_cpu, initial_memory)

            save_model(model, args.model_file)

            # Convert input example to float to avoid MLflow schema warning
            input_example = pd.DataFrame(X_train[:1]).astype(float)

            # Infer model signature with data converted to float
            signature = infer_signature(X_train.astype(float), model.predict(X_train))

            # Log the model with MLflow
            mlflow.sklearn.log_model(
                model,
                "random_forest_model",
                input_example=input_example,
                signature=signature
            )

            # Register the model in the MLflow Model Registry
            model_uri = f"runs:/{run.info.run_id}/random_forest_model"
            mlflow.register_model(model_uri, "ChurnPredictionModel")

            print("✅ Model training complete. Model saved and registered.")

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

            # Log evaluation metrics
            print("🔹 Logging evaluation metrics to MLflow...")
            mlflow.log_metric("accuracy", report["accuracy"])
            mlflow.log_metric("precision_True", report["True"]["precision"])
            mlflow.log_metric("recall_True", report["True"]["recall"])
            mlflow.log_metric("f1_True", report["True"]["f1-score"])
            mlflow.log_metric("precision_False", report["False"]["precision"])
            mlflow.log_metric("recall_False", report["False"]["recall"])
            mlflow.log_metric("f1_False", report["False"]["f1-score"])
            print("✅ Metrics logged successfully.")

            # Log inference metrics
            log_inference_metrics(model, X_test)

            print("✅ Evaluation complete. Metrics logged in MLflow.")

if __name__ == "__main__":
    main()
