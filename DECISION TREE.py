Python 3.12.4 (tags/v3.12.4:8e8a4ba, Jun  6 2024, 19:30:16) [MSC v.1940 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.  

# ==========================================
# DECISION TREE CLASSIFICATION
# ==========================================

# 1. Import libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree

import matplotlib.pyplot as plt


# ==========================================
# 2. Load dataset
# ==========================================

df = pd.read_csv("data.csv")

print(df.head())
print(df.info())


# ==========================================
# 3. Separate X and y
# ==========================================

# X = input/features
# y = target/output

X = df.drop("target", axis=1)
y = df["target"]

print("X:")
print(X.head())

print("y:")
print(y.head())


# ==========================================
# 4. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# ==========================================
# 5. Create Decision Tree model
# ==========================================

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)


# ==========================================
# 6. Train the model
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# 7. Make predictions
# ==========================================

y_pred = model.predict(X_test)

print("Predictions:")
print(y_pred)


# ==========================================
# 8. Evaluate model
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)


# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# ==========================================
# 9. Predict a new data point
# ==========================================

new_data = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_data)

print("Prediction:", prediction)


# ==========================================
# 10. Visualize Decision Tree
# ==========================================

plt.figure(figsize=(15, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=[str(c) for c in model.classes_],
    filled=True
)

plt.show()
