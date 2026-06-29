"""
Model monitoring using Evidently AI.
Generates data drift and model performance reports.
"""
import pickle
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently import ColumnMapping

from src.preprocessing import load_and_prepare_data
from src import config


def load_artifacts():
    with open(config.VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(config.RF_MODEL_PATH, "rb") as f:
        rf_model = pickle.load(f)
    return vectorizer, rf_model


def run_monitoring():
    print("Loading data...")
    df = load_and_prepare_data()

    print("Loading model and vectorizer...")
    vectorizer, rf_model = load_artifacts()

    # Split into reference (70%) and current (30%) data
    split = int(len(df) * 0.7)
    reference = df.iloc[:split].copy()
    current = df.iloc[split:].copy()

    print(f"Reference data: {len(reference)} rows")
    print(f"Current data:   {len(current)} rows")

    # Generate predictions
    ref_vectorized = vectorizer.transform(reference["lemmatized_tweet"])
    cur_vectorized = vectorizer.transform(current["lemmatized_tweet"])

    reference["prediction"] = rf_model.predict(ref_vectorized)
    current["prediction"] = rf_model.predict(cur_vectorized)

    # Keep only relevant columns
    reference = reference[["label", "prediction"]]
    current = current[["label", "prediction"]]

    # Column mapping for Evidently
    column_mapping = ColumnMapping(
        target="label",
        prediction="prediction",
    )

    # Generate Data Drift + Classification Performance report
    print("Generating monitoring report...")
    report = Report(metrics=[
        DataDriftPreset(),
        ClassificationPreset(),
    ])

    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )

    report.save_html("monitoring_report.html")
    print("Report saved to monitoring_report.html — open it in your browser.")


if __name__ == "__main__":
    run_monitoring()
