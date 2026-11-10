import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
churn_df = pd.DataFrame({
    "monthly_usage": [20, 80, 15, 70, 25, 90, 30, 75, 18, 85, 40, 65],
    "support_tickets": [5, 1, 6, 1, 4, 0, 5, 1, 7, 1, 3, 2],
    "tenure_months": [3, 24, 2, 30, 5, 36, 4, 28, 2, 32, 12, 20],
    "cancellation_date": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
    "churn": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]
})

#print(churn_df)

'''X_leak = churn_df[
    ["monthly_usage", "support_tickets", "tenure_months", "cancellation_date"]
]
y_churn = churn_df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X_leak, y_churn, test_size=0.25, random_state=42, stratify=y_churn
)

leak_model = LogisticRegression(max_iter=1000)
leak_model.fit(X_train, y_train)
leak_pred = leak_model.predict(X_test)

print("Accuracy with leakage:", accuracy_score(y_test, leak_pred))'''

X_clean = churn_df[
    ["monthly_usage", "support_tickets", "tenure_months"]
]
y_churn = churn_df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_churn, test_size=0.25, random_state=42, stratify=y_churn
)

clean_model = LogisticRegression(max_iter=1000)
clean_model.fit(X_train, y_train)
clean_pred = clean_model.predict(X_test)

print("Accuracy without leakage:", accuracy_score(y_test, clean_pred))


# Train a Churn Classifier
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)

probabilities = classifier.predict_proba(X_test)[:, 1]  
predictions = classifier.predict(X_test)

print("Classes:", predictions)
print("Probabilities:", probabilities)

# 15. Evaluate Classification
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions, zero_division=0))
print("Recall:", recall_score(y_test, predictions, zero_division=0))
print("F1:", f1_score(y_test, predictions, zero_division=0))
print("Confusion Matrix:",confusion_matrix(y_test, predictions))

# 16. Understand the Confusion Matrix
# • TP: predicted churn and actually churned.
# • FP: predicted churn but did not churn.
# • TN: predicted no churn and did not churn.
# • FN: predicted no churn but actually churned.

# 17. Experiment With the Classification Threshold
threshold = 0.30
custom_predictions = (probabilities >= threshold).astype(int)

print("Threshold:", threshold)
print("Precision:", precision_score(y_test, custom_predictions, zero_division=0))
print("Recall:", recall_score(y_test, custom_predictions, zero_division=0))





