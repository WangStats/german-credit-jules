# Model Evaluation in a Banking Context

## Metrics Overview

The following metrics were calculated for the Random Forest model trained to predict credit risk:

*   **Accuracy**: The overall percentage of correct predictions.
*   **Precision**: Of all the applicants predicted as 'Bad' risk, how many were actually 'Bad'.
*   **Recall (Sensitivity)**: Of all the actual 'Bad' risk applicants, how many did the model correctly identify.
*   **F1-Score**: The harmonic mean of Precision and Recall.
*   **ROC-AUC**: The Area Under the Receiver Operating Characteristic Curve, representing the model's ability to distinguish between classes.

## Judging "Good" vs "Bad" Models

In a banking/credit context, the cost of misclassification is often asymmetric.

1.  **False Negatives (Type II Error)**: Predicting a 'Bad' risk applicant as 'Good'.
    *   **Consequence**: The bank approves a loan that defaults.
    *   **Cost**: Loss of principal amount, interest, and recovery costs. This is typically **very high**.

2.  **False Positives (Type I Error)**: Predicting a 'Good' risk applicant as 'Bad'.
    *   **Consequence**: The bank rejects a good customer.
    *   **Cost**: Loss of potential interest income (opportunity cost) and potential damage to customer reputation. This is typically **lower** than the cost of a default.

### Critical Metric: Recall (for the 'Bad' Class)

Because missing a defaulter (False Negative) is expensive, a "good" model in this context should prioritize **Recall**.

*   **High Recall**: Means we catch most of the bad applicants. We might flag some good ones as bad (lower Precision), but we avoid the heavy losses of defaults.
*   **Trade-off**: Increasing Recall often decreases Precision. The bank must decide the acceptable threshold based on their specific Cost Matrix (e.g., if a default costs 5x more than a lost customer, we tolerate more False Positives to catch Defaults).

### ROC-AUC

A good AUC score (closer to 1.0) indicates the model is robust at separating good and bad risks across different thresholds. It helps in selecting the optimal decision threshold to balance the trade-off mentioned above.

## Conclusion

To evaluate this model:
*   Look beyond Accuracy (especially if classes are imbalanced).
*   Focus on **Recall** for the 'Bad' class (Class 1).
*   Use the **Confusion Matrix** to see raw counts of False Negatives vs False Positives.
*   A model with 90% accuracy but 20% recall is likely **bad** for this use case. A model with 70% accuracy but 85% recall might be **better** financially.
