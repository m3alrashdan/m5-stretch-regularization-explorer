# Regularization Explorer — Logistic Regression

This project explores how L1 and L2 regularization affect model coefficients
in a Logistic Regression model using a telecom churn dataset.

## Objective

To visualize how feature coefficients change as regularization strength varies
and understand the differences between L1 (Lasso) and L2 (Ridge).

---

## Methodology

- Generated 20 values of C using logarithmic spacing from 0.001 to 100
- Trained Logistic Regression models using:
  - L1 regularization (sparse solutions)
  - L2 regularization (smooth shrinkage)
- Standardized all features before training
- Tracked coefficient values across all models
- Visualized regularization paths for top features

---

## Results

The plots show how coefficients behave under different regularization strengths:

- **L1 Regularization** drives many coefficients exactly to zero,
  performing implicit feature selection.
- **L2 Regularization** shrinks coefficients smoothly toward zero,
  but rarely eliminates them completely.

---

## Interpretation

The regularization paths clearly show the effect of strong regularization on model coefficients.
At very small values of C (e.g., 0.001), L1 regularization forces all feature coefficients to zero,
indicating that the model is heavily penalized and unable to retain any predictive signal.

As C increases, coefficients begin to reappear, showing which features are more robust and
less sensitive to regularization. Features that remain non-zero at lower regularization strengths
are likely stronger predictors.

In contrast, L2 regularization shrinks coefficients smoothly toward zero but does not eliminate
them entirely. This results in a more stable model that retains all features but reduces their impact.

Based on this behavior, L1 is useful for feature selection and identifying the most important
variables, while L2 is preferable when we want a more stable model that handles correlated
features without completely removing them.

Additionally, the `customer_id` column was removed as it is a non-informative identifier.

---

## Files

- `regularization_explorer.py` → Main script
- `regularization_path.png` → Visualization output
- `telecom_churn.csv` → Dataset

---

## How to Run

```bash
pip install -r requirements.txt
python regularization_explorer.py
```
