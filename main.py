"""
Multi-Model Comparison for Disease Prediction
---------------------------------------------
This script compares multiple ML classifiers on the Heart Attack dataset
and identifies the best-performing one based on test accuracy and F1-score.
It also saves the best model and scaler for later use.
"""

import logging
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Import models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

# ---------------- LOGGING CONFIG ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------- LOAD DATA ----------------
def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from Excel file."""
    data = pd.read_excel(filepath)
    logger.info(f"✅ Dataset loaded successfully with shape {data.shape}.")
    return data


def preprocess_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features and target variable."""
    if 'target' not in data.columns:
        raise ValueError("'target' column not found in dataset.")
    X = data.drop(columns=['target'])
    y = data['target']
    return X, y


def split_and_scale(X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Split and standardize data."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ---------------- MODEL TRAINING ----------------
def get_models() -> Dict[str, Any]:
    """Return a dictionary of ML models to compare."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM (RBF Kernel)": SVC(kernel="rbf", probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": xgb.XGBClassifier(
            use_label_encoder=False, eval_metric='logloss', random_state=42
        )
    }


def evaluate_model(model, X_train, y_train, X_test, y_test) -> Dict[str, float]:
    """Train and evaluate a single model."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Accuracy: {acc:.3f}, F1-score: {f1:.3f}")
    logger.debug(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    logger.debug(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    return {"accuracy": acc, "f1_score": f1}


def compare_models(filepath: str) -> None:
    """Main comparison routine."""
    data = load_data(filepath)
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    models = get_models()
    results = {}
    trained_models = {}

    logger.info("🚀 Starting model training and evaluation...")

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results[name] = {"accuracy": acc, "f1_score": f1}
        trained_models[name] = model

    # Rank models
    results_df = pd.DataFrame(results).T.sort_values(by="accuracy", ascending=False)
    logger.info("\n📊 Model Performance Summary:")
    print(results_df)

    # Identify and save the best model
    best_model_name = results_df.index[0]
    best_model = trained_models[best_model_name]
    logger.info(f"\n🏆 Best Model: {best_model_name}")
    logger.info(f"✅ Accuracy: {results_df.iloc[0]['accuracy']:.3f}, F1: {results_df.iloc[0]['f1_score']:.3f}")

    # Save best model and scaler
    joblib.dump(best_model, "Models/best_model.pkl")
    joblib.dump(scaler, "Models/scaler.pkl")
    logger.info(f"💾 Saved best model as 'Models/best_model.pkl' and scaler as 'Models/scaler.pkl'.")


if __name__ == "__main__":
    compare_models("Data/Heart_Attack.xlsx")
