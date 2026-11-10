import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

X, y = make_classification(
    n_samples=2000,
    n_features=6,
    n_informative=4,
    n_redundant=1,
    weights=[0.90, 0.10],
    random_state=42
)

df = pd.DataFrame(X, columns=[
    "distance", "weather_risk", "warehouse_load",
    "carrier_risk", "order_priority", "route_risk"
])

df["delayed"] = y

print(df.head())
print(df["delayed"].value_counts())

X = df.drop("delayed", axis=1)
y = df["delayed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

print("Training rows:", len(X_train))
print("Test rows:", len(X_test))

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("Model trained successfully.") #algorithm that makes predictions by combining many decision trees and taking a majority vote

y_pred = model.predict(X_test)

print(y_pred[:20])

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy : {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")

cm = confusion_matrix(y_test, y_pred)
print(cm)

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Shipment Delay Confusion Matrix")
plt.show()

bad_predictions = np.zeros(len(y_test))

bad_accuracy = accuracy_score(y_test, bad_predictions)
bad_precision = precision_score(y_test, bad_predictions, zero_division=0)
bad_recall = recall_score(y_test, bad_predictions, zero_division=0)

print(f"Bad model accuracy : {bad_accuracy:.3f}")
print(f"Bad model precision: {bad_precision:.3f}")
print(f"Bad model recall   : {bad_recall:.3f}")

y_prob = model.predict_proba(X_test)[:, 1]

print(y_prob[:10])

def evaluate_threshold(threshold):
    predictions = (y_prob >= threshold).astype(int)

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0)
    }

results = []

for threshold in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
    results.append(evaluate_threshold(threshold))

threshold_df = pd.DataFrame(results)
print(threshold_df.round(3))

plt.figure(figsize=(8, 5))

plt.plot(threshold_df["threshold"], threshold_df["precision"],
         marker="o", label="Precision")

plt.plot(threshold_df["threshold"], threshold_df["recall"],
         marker="o", label="Recall")

plt.plot(threshold_df["threshold"], threshold_df["accuracy"],
         marker="o", label="Accuracy")

plt.xlabel("Classification Threshold")
plt.ylabel("Metric")
plt.title("Metrics vs Classification Threshold")
plt.legend()
plt.grid(True)
plt.show()

target_recall = 0.80

eligible = threshold_df[
    threshold_df["recall"] >= target_recall
]

print(eligible.round(3))

comparison = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall"],
    "Trained Model": [accuracy, precision, recall],
    "Always On-Time Model": [
        bad_accuracy, bad_precision, bad_recall
    ]
})

print(comparison.round(3))

for threshold in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
    predictions = (y_prob >= threshold).astype(int)
    f1 = f1_score(y_test, predictions, zero_division=0)
    print(f"Threshold={threshold:.2f}  F1={f1:.3f}")
