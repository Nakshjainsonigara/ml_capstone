"""Simple preprocessing, training, and MLflow tracking for the Iris dataset."""
from __future__ import annotations

import argparse
import pathlib
import tempfile

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Logistic Regression on Iris dataset")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion for test split")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max-iter", type=int, default=500, help="Max iterations for optimizer")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength")
    parser.add_argument(
        "--solver",
        type=str,
        default="lbfgs",
        choices=["lbfgs", "saga", "liblinear", "newton-cg", "sag"],
        help="Optimization algorithm",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="iris_classification",
        help="MLflow experiment name",
    )
    return parser.parse_args()


def load_data(data_path: pathlib.Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(data_path)
    features = df.drop(columns=["species"])
    target = df["species"]
    return features, target


def build_pipeline(C: float, max_iter: int, solver: str, random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=C,
                    multi_class="auto",
                    solver=solver,
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            ),
        ]
    )


def main() -> None:
    args = parse_args()

    project_root = pathlib.Path(__file__).resolve().parent.parent
    data_file = project_root / "data" / "iris.csv"

    mlflow.set_tracking_uri(f"file://{project_root / 'mlruns'}")
    mlflow.set_experiment(args.experiment_name)

    X, y = load_data(data_file)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    n_samples = len(y_encoded)
    n_classes = len(set(y_encoded))

    test_size = args.test_size
    if test_size < 1:
        min_fraction = n_classes / n_samples
        if test_size < min_fraction:
            adjusted = min(0.5, max(min_fraction, test_size))
            print(
                f"Adjusting test_size from {test_size:.2f} to {adjusted:.2f} "
                "to ensure each class appears in the test split."
            )
            test_size = adjusted
    else:
        min_count = n_classes
        if test_size < min_count:
            adjusted = min_count
            print(
                f"Adjusting test_size from {test_size} to {adjusted} "
                "to ensure each class appears in the test split."
            )
            test_size = adjusted

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=args.random_state,
        stratify=y_encoded,
    )

    pipeline = build_pipeline(args.C, args.max_iter, args.solver, args.random_state)

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "solver": args.solver,
                "max_iter": args.max_iter,
                "C": args.C,
                "test_size": args.test_size,
                "random_state": args.random_state,
            }
        )

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        mlflow.log_metrics({"accuracy": acc, "mse": mse})

        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        mlflow.log_dict({"classes": label_encoder.classes_.tolist()}, "model/label_encoder_classes.json")

        report = classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            digits=4,
        )

        with tempfile.NamedTemporaryFile("w", suffix="_classification_report.txt", delete=False) as tmp:
            tmp.write(report)
            report_path = pathlib.Path(tmp.name)
        mlflow.log_artifact(report_path, artifact_path="evaluation")
        report_path.unlink(missing_ok=True)

        print("Run ID:", run.info.run_id)
        print("Classification report:\n", report)
        print("MLflow artifacts saved to:", run.info.artifact_uri)


if __name__ == "__main__":
    main()
