import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, ConfusionMatrixDisplay
)
import os

def train_and_evaluate():
    # Load data
    df = pd.read_csv('german_credit_data.csv')

    # Define Target and Features
    # 'Risk' is the target. Map 'bad' -> 1, 'good' -> 0
    # This makes 'bad' the positive class, which is standard for risk detection.
    y = df['Risk'].map({'bad': 1, 'good': 0})
    X = df.drop('Risk', axis=1)

    # Identify Categorical and Numerical columns
    # Numerical: Age, Credit amount, Duration
    # Categorical: Sex, Job, Housing, Saving accounts, Checking account, Purpose

    numerical_cols = ['Age', 'Credit amount', 'Duration']
    categorical_cols = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']

    # Preprocessing Pipeline
    # Numeric: Scale (StandardScaler)
    # Categorical: One-Hot Encode (handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    # Model Pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train Model
    print("Training Random Forest Classifier...")
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Probability of class 1 (bad)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print("\nModel Evaluation Metrics:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    # Ensure plots directory exists
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Visual Evaluation: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Good', 'Bad'])

    plt.figure(figsize=(8, 6))
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix (Random Forest)')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png')
    plt.close() # Close figure to free memory
    # Note: ConfusionMatrixDisplay.plot() creates its own figure if ax is not provided,
    # but calling plt.figure before helps control size if we passed ax.
    # Actually disp.plot() returns the display object.
    # To be safe with saving, let's do it explicitly on an axes.
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title('Confusion Matrix (Random Forest)')
    fig.savefig('plots/confusion_matrix.png')
    plt.close(fig)


    # Visual Evaluation: ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('plots/roc_curve.png')
    plt.close()

    print("\nPlots saved to plots/confusion_matrix.png and plots/roc_curve.png")

if __name__ == "__main__":
    train_and_evaluate()
