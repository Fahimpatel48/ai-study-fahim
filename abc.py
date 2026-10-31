import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# =========================================================
# 1. LOAD DATASET
# =========================================================

df = pd.read_csv("dataset/Railway Ticket Confirmation.csv")

print("\n========== DATASET ==========")
print("Total Rows:", len(df))
print("Total Columns:", len(df.columns))

print("\nColumns:")
print(df.columns.tolist())


# =========================================================
# 2. HANDLE MISSING VALUES
# =========================================================

df["Special Considerations"] = df["Special Considerations"].fillna("None")


# =========================================================
# 3. LABEL ENCODING
# =========================================================

le_from = LabelEncoder()
le_To = LabelEncoder()
le_Class = LabelEncoder()
le_Quota = LabelEncoder()
le_Status = LabelEncoder()
le_Special = LabelEncoder()
le_target = LabelEncoder()


df["From"] = le_from.fit_transform(df["From"])

df["To"] = le_To.fit_transform(df["To"])

df["Class of Travel"] = le_Class.fit_transform(
    df["Class of Travel"]
)

df["Quota"] = le_Quota.fit_transform(
    df["Quota"]
)

df["Current Status"] = le_Status.fit_transform(
    df["Current Status"]
)

df["Special Considerations"] = le_Special.fit_transform(
    df["Special Considerations"]
)

df["Confirmation Status"] = le_target.fit_transform(
    df["Confirmation Status"]
)


# =========================================================
# 4. SELECT FEATURES
# =========================================================

x = df[
    [
        "From",
        "To",
        "Class of Travel",
        "Quota",
        "Current Status",
        "Number of Passengers",
        "Special Considerations"
    ]
]

y = df["Confirmation Status"]


# =========================================================
# 5. TRAIN TEST SPLIT
# =========================================================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\n========== DATA SPLIT ==========")
print("Training Data:", x_train.shape)
print("Testing Data :", x_test.shape)


# =========================================================
# 6. LOGISTIC REGRESSION MODEL
# =========================================================

model = LogisticRegression(
    max_iter=1000
)

model.fit(x_train, y_train)


# =========================================================
# 7. PREDICTION
# =========================================================

y_pred = model.predict(x_test)

# Probability of positive class
y_probability = model.predict_proba(x_test)[:, 1]


# =========================================================
# 8. MODEL PERFORMANCE
# =========================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print("\n========== MODEL PERFORMANCE ==========")

print("Accuracy  :", round(accuracy * 100, 2), "%")
print("Precision :", round(precision * 100, 2), "%")
print("Recall    :", round(recall * 100, 2), "%")
print("F1 Score  :", round(f1 * 100, 2), "%")


# =========================================================
# 9. TARGET CLASSES
# =========================================================

print("\n========== TARGET CLASSES ==========")
print(le_target.classes_)


# =========================================================
# 10. PLOT 1 - CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=le_target.classes_
)

disp.plot()

plt.title("Logistic Regression - Confusion Matrix")

plt.tight_layout()

plt.show()


# =========================================================
# 11. PLOT 2 - ACTUAL VS PREDICTED
# =========================================================

plt.figure(figsize=(10, 5))

# First 100 records
number_of_records = min(100, len(y_test))

actual_values = y_test.iloc[:number_of_records].values
predicted_values = y_pred[:number_of_records]

plt.plot(
    range(number_of_records),
    actual_values,
    "o",
    label="Actual"
)

plt.plot(
    range(number_of_records),
    predicted_values,
    "x",
    label="Predicted"
)

plt.xlabel("Test Record")

plt.ylabel("Confirmation Status")

plt.title(
    "Actual vs Predicted - Logistic Regression"
)

plt.yticks(
    [0, 1],
    ["Not Confirmed", "Confirmed"]
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# =========================================================
# 12. PLOT 3 - CONFIRMATION PROBABILITY
# =========================================================

plt.figure(figsize=(8, 5))

plt.hist(
    y_probability * 100,
    bins=10,
    edgecolor="black"
)

plt.xlabel("Confirmation Probability (%)")

plt.ylabel("Number of Tickets")

plt.title(
    "Railway Ticket Confirmation Probability"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# =========================================================
# 13. PLOT 4 - FEATURE COEFFICIENTS
# =========================================================

feature_names = [
    "From",
    "To",
    "Class of Travel",
    "Quota",
    "Current Status",
    "Number of Passengers",
    "Special Considerations"
]

coefficients = model.coef_[0]


plt.figure(figsize=(10, 6))

plt.barh(
    feature_names,
    coefficients
)

plt.xlabel(
    "Logistic Regression Coefficient"
)

plt.ylabel("Features")

plt.title(
    "Feature Coefficients - Logistic Regression"
)

plt.axvline(
    0,
    linewidth=1
)

plt.grid(True)

plt.tight_layout()

plt.show()


# =========================================================
# 14. USER INPUT
# =========================================================

print("\n========================================")
print("        RAILWAY TICKET PREDICTION")
print("========================================")


# FROM

print("\nFROM:")
print(le_from.classes_)

from_input = input("From: ")


# TO

print("\nTO:")
print(le_To.classes_)

to_input = input("TO: ")


# CLASS

print("\nCLASS OF TRAVEL:")
print(le_Class.classes_)

class_input = input(
    "Class of Travel: "
)


# QUOTA

print("\nQUOTA:")
print(le_Quota.classes_)

quota_input = input(
    "Quota: "
)


# CURRENT STATUS

print("\nCURRENT STATUS:")
print(le_Status.classes_)

status_input = input(
    "Current Status: "
)


# NUMBER OF PASSENGERS

passengers_input = int(
    input("Number of Passengers: ")
)


# SPECIAL CONSIDERATION

print("\nSPECIAL CONSIDERATIONS:")
print(le_Special.classes_)

special_input = input(
    "Special Consideration: "
)


# =========================================================
# 15. CONVERT USER INPUT INTO NUMBERS
# =========================================================

if special_input.lower() == "none":
    special_input = "None"


from_value = le_from.transform(
    [from_input]
)[0]


to_value = le_To.transform(
    [to_input]
)[0]


class_value = le_Class.transform(
    [class_input]
)[0]


quota_value = le_Quota.transform(
    [quota_input]
)[0]


status_value = le_Status.transform(
    [status_input]
)[0]


special_value = le_Special.transform(
    [special_input]
)[0]


# =========================================================
# 16. NEW TICKET DATA
# =========================================================

new_ticket = np.array(
    [[
        from_value,
        to_value,
        class_value,
        quota_value,
        status_value,
        passengers_input,
        special_value
    ]]
)


# =========================================================
# 17. FINAL PREDICTION
# =========================================================

prediction = model.predict(
    new_ticket
)


probability = model.predict_proba(
    new_ticket
)


# Probability of Confirmed class
confirmation_percentage = probability[0][1] * 100


# =========================================================
# 18. FINAL RESULT
# =========================================================

print("\n========================================")
print("             PREDICTION RESULT")
print("========================================")


if prediction[0] == 1:

    print("Confirmation Status : CONFIRMED")

else:

    print("Confirmation Status : NOT CONFIRMED")


print(
    "Confirmation Chance :",
    round(confirmation_percentage, 2),
    "%"
)


print("========================================")