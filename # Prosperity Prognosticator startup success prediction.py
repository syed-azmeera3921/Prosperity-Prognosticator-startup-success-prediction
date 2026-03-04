# Prosperity Prognosticator
# Machine Learning for Startup Success Prediction
# Author: Syed Azmeera Begum

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. Data Collection (Sample Dataset)
# -----------------------------

data = {
    "Funding": [50000, 200000, 75000, 300000, 150000, 400000, 100000, 250000],
    "Team_Experience": [2, 5, 3, 7, 4, 8, 3, 6],
    "Market_Size": [1000000, 5000000, 2000000, 8000000, 4000000, 9000000, 2500000, 7000000],
    "Success": [0, 1, 0, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

print("\n===== DATASET =====")
print(df)

# -----------------------------
# 2. Exploratory Data Analysis
# -----------------------------

print("\n===== DATA DESCRIPTION =====")
print(df.describe())

# Correlation Matrix
plt.figure()
plt.matshow(df.corr())
plt.title("Correlation Matrix")
plt.colorbar()
plt.show()

# -----------------------------
# 3. Data Preparation
# -----------------------------

X = df[["Funding", "Team_Experience", "Market_Size"]]
y = df["Success"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Model Building
# -----------------------------

model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# 5. Performance Testing
# -----------------------------

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n===== MODEL ACCURACY =====")
print("Accuracy:", accuracy)

# -----------------------------
# 6. User Prediction
# -----------------------------

print("\n===== PREDICT STARTUP SUCCESS =====")

funding = float(input("Enter Funding Amount: "))
experience = float(input("Enter Team Experience (Years): "))
market = float(input("Enter Market Size: "))

input_data = np.array([[funding, experience, market]])
prediction = model.predict(input_data)

if prediction[0] == 1:
    print("Prediction: Startup Likely to Succeed 🚀")
else:
    print("Prediction: Startup May Fail ⚠️")