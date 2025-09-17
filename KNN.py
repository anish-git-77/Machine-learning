import pandas as pd
import math
# Load the dataset
df = pd.read_csv(r'D:/ml_gopal/StressLevelDataset.csv')
print(df)
# Target variable: Stress Level
y = df["stress_level"]
print(y)
# Feature variables
x1 = df["anxiety_level"]
x2 = df["mental_health_history"]
x3 = df["headache"]
# Initialize list to store the distances
l = []
# Assume a new sample with some given features (a, b, c)
a = 3  # Example anxiety level of new data point
b = 1  # Example mental health history of new data point
c = 1  # Example headache level of new data point
# Compute the Euclidean distance for all data points
for i in range(0, len(df)):  # Loop through all rows in the dataset
    k1 = (x1[i] - a)**2
    k2 = (x2[i] - b)**2
    k3 = (x3[i] - c)**2
    k = k1 + k2 + k3  # Sum of squared differences
    s = math.sqrt(k)  # Euclidean distance
    l.append((s, y[i]))  # Append the distance and corresponding stress level
# Sort the distances list by the Euclidean distance (smallest to largest)
l.sort(key=lambda x: x[0])

# Set K (number of nearest neighbors)
K = 3
# Initialize counters for each stress level
c1 = 0  # Counter for stress_level = 0
c2 = 0  # Counter for stress_level = 1
c3 = 0  # Counter for stress_level = 2
# Get the K nearest neighbors and count the occurrences of each stress level
for i in range(K):
    dist, stress = l[i]
    if stress == 0:
        c1 += 1
    elif stress == 1:
        c2 += 1
    elif stress == 2:
        c3 += 1
# Output the counts of each class among the K nearest neighbors
print(f"Stress Level 0 count: {c1}")
print(f"Stress Level 1 count: {c2}")
print(f"Stress Level 2 count: {c3}")
# Determine the predicted class (majority vote)
if c1 > c2 and c1 > c3:
    predicted_stress_level = 0
elif c2 > c1 and c2 > c3:
    predicted_stress_level = 1
else:
    predicted_stress_level = 2
# Output the predicted stress level
print(f"Predicted Stress Level: {predicted_stress_level}")
