"""
Regularization Explorer — Robust Version (Handles column issues)

This version automatically detects the target column and cleans data.
"""

# =========================
# 1. Imports
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# =========================
# 2. Load Data
# =========================
df = pd.read_csv("telecom_churn.csv")

# Clean column names (VERY IMPORTANT)
df.columns = df.columns.str.strip().str.lower()

print("Columns in dataset:\n", df.columns.tolist())


# =========================
# 3. Detect Target Column Automatically
# =========================
target_col = None

for col in df.columns:
    if "churn" in col:
        target_col = col
        break

if target_col is None:
    raise ValueError("No churn column found. Check dataset column names.")

print(f"\nUsing target column: {target_col}")


# =========================
# 4. Prepare Target
# =========================
y = df[target_col]

# Convert Yes/No to 1/0 if needed
if y.dtype == "object":
    y = y.str.strip().str.lower()
    y = y.map({"yes": 1, "no": 0})

# Drop target from features
# Drop target + ID column if exists
drop_cols = [target_col]

if "customer_id" in df.columns:
    drop_cols.append("customer_id")

X = df.drop(columns=drop_cols)

# =========================
# 5. Preprocessing
# =========================

# One-hot encoding for categorical variables
X = pd.get_dummies(X, drop_first=True)

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

feature_names = X.columns


# =========================
# 6. Generate C values
# =========================
C_values = np.logspace(-3, 2, 20)


# =========================
# 7. Train Models
# =========================
coeffs_l1 = []
coeffs_l2 = []

for C in C_values:
    # L1
    model_l1 = LogisticRegression(
        penalty='l1',
        solver='saga',
        C=C,
        max_iter=5000
    )
    model_l1.fit(X_scaled, y)
    coeffs_l1.append(model_l1.coef_[0])

    # L2
    model_l2 = LogisticRegression(
        penalty='l2',
        solver='saga',
        C=C,
        max_iter=5000
    )
    model_l2.fit(X_scaled, y)
    coeffs_l2.append(model_l2.coef_[0])

coeffs_l1 = np.array(coeffs_l1)
coeffs_l2 = np.array(coeffs_l2)


# =========================
# 8. Select Top Features
# =========================
final_coeffs = np.abs(coeffs_l2[-1])

top_k = 10
top_indices = np.argsort(final_coeffs)[-top_k:]


# =========================
# 9. Plot (Clean + Professional)
# =========================
plt.figure(figsize=(16, 6))

# L1 Plot
plt.subplot(1, 2, 1)
for i in top_indices:
    plt.plot(C_values, coeffs_l1[:, i], label=feature_names[i])

plt.xscale("log")
plt.title("L1 Regularization Path (Sparse)", fontsize=12)
plt.xlabel("C (log scale)")
plt.ylabel("Coefficient Value")
plt.axhline(0)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)

# L2 Plot
plt.subplot(1, 2, 2)
for i in top_indices:
    plt.plot(C_values, coeffs_l2[:, i], label=feature_names[i])

plt.xscale("log")
plt.title("L2 Regularization Path (Smooth)", fontsize=12)
plt.xlabel("C (log scale)")
plt.axhline(0)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)

plt.tight_layout()
plt.savefig("regularization_path.png", dpi=300, bbox_inches='tight')
plt.show()


# =========================
# 10. Features Eliminated (L1)
# =========================
zero_tracking = []

for i, feature in enumerate(feature_names):
    for j, coef in enumerate(coeffs_l1[:, i]):
        if coef == 0:
            zero_tracking.append((feature, C_values[j]))
            break

zero_tracking = sorted(zero_tracking, key=lambda x: x[1])

print("\nTop features eliminated first under L1:\n")

for feature, c in zero_tracking[:10]:
    print(f"{feature} -> C = {c:.5f}")