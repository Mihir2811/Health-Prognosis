"""
Disease Prediction using Logistic Regression
Trains a Logistic Regression model to predict heart disease likelihood from patient data.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Tuple, Any


# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from an Excel file."""
    try:
        data = pd.read_excel(filepath)
        logger.info("✅ Dataset loaded successfully.")
        logger.info(f"Dataset shape: {data.shape}")
        missing_values = data.isnull().sum()
        logger.info(f"Missing values per column:\n{missing_values}")
        return data
    except FileNotFoundError:
        logger.error(f"❌ File not found at path: {filepath}")
        raise
    except Exception as e:
        logger.error(f"❌ Error reading the file: {e}")
        raise


def preprocess_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features and target variable."""
    if 'target' not in data.columns:
        raise ValueError("'target' column not found in dataset.")
    
    X = data.drop(columns=['target'])
    y = data['target']
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataset into training and testing sets with stratification."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    logger.info(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """Train logistic regression model."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    logger.info("✅ Model training completed.")
    return model


def evaluate_model(model: LogisticRegression, X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float, float]:
    """Evaluate model performance on both training and test datasets."""
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    logger.info(f"🔹 Training Accuracy: {train_acc:.3f}")
    logger.info(f"🔹 Test Accuracy: {test_acc:.3f}")

    # Optional detailed report
    logger.debug("\nClassification Report (Test):\n" + classification_report(y_test, test_pred))
    logger.debug("\nConfusion Matrix (Test):\n" + str(confusion_matrix(y_test, test_pred)))

    return train_acc, test_acc


def predict_heart_disease(model: LogisticRegression, input_data: Tuple[Any, ...]) -> int:
    """
    Predict whether a patient has heart disease based on input features.
    
    Parameters:
        model (LogisticRegression): Trained model.
        input_data (tuple): Patient feature values.
        
    Returns:
        int: Prediction result (0 or 1).
    """
    try:
        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        result = "has heart disease" if prediction == 1 else "does NOT have heart disease"
        logger.info(f"Prediction: The person {result}.")
        return prediction
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise


def main() -> None:
    """Main execution flow."""
    filepath = "Heart_Attack.xlsx"

    # Load and prepare data
    data = load_data(filepath)
    X, y = preprocess_data(data)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_train, y_train, X_test, y_test)

    # Make example prediction
    example_input = (63, 1, 3, 150, 268, 1, 1, 187, 0, 3.6, 0, 2, 2)
    predict_heart_disease(model, example_input)


if __name__ == "__main__":
    main()
