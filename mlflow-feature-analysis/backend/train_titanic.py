# backend/train_titanic.py
"""
Training script for Titanic dataset
Logs model to MLflow for use in Feature Analysis dashboard
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MLFLOW_TRACKING_URI = "http://localhost:5000"
DATA_PATH = "./data/titanic.csv"  # Replace with your path

def load_and_preprocess_data(filepath):
    """Load and preprocess Titanic dataset"""
    logger.info("Loading Titanic dataset...")
    
    df = pd.read_csv(filepath)
    
    # Drop unnecessary columns
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    
    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Encode categorical variables
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    
    df['Sex'] = le_sex.fit_transform(df['Sex'])
    df['Embarked'] = le_embarked.fit_transform(df['Embarked'])
    
    # Separate features and target
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Features: {X.columns.tolist()}")
    
    return X, y

def train_and_log_model(X, y, model_type='logistic', run_name='titanic_baseline'):
    """Train model and log to MLflow"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Select model
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000, random_state=42)
        model_name = "logistic_regression"
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model_name = "random_forest"
        X_train_scaled = X_train  # RF doesn't need scaling
        X_test_scaled = X_test
    
    # Train
    logger.info(f"Training {model_name}...")
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logger.info(f"Model Performance:")
    logger.info(f"  Accuracy:  {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1-Score:  {f1:.4f}")
    
    # Log to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        
        if model_type == 'rf':
            mlflow.log_param("n_estimators", 100)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Log confusion matrix as artifact
        cm = confusion_matrix(y_test, y_pred)
        np.save("confusion_matrix.npy", cm)
        mlflow.log_artifact("confusion_matrix.npy")
        
        # Log test data for later SHAP computation
        test_data = pd.DataFrame(X_test_scaled, columns=X.columns)
        test_data.to_csv("test_data.csv", index=False)
        mlflow.log_artifact("test_data.csv")
        
        logger.info(f"Model logged to MLflow with run_id: {mlflow.active_run().info.run_id}")

def main():
    """Main execution"""
    try:
        # Load and preprocess
        X, y = load_and_preprocess_data(DATA_PATH)
        
        # Train Logistic Regression
        train_and_log_model(X, y, model_type='logistic', run_name='titanic_logistic')
        
        # Train Random Forest
        train_and_log_model(X, y, model_type='rf', run_name='titanic_random_forest')
        
        logger.info("Training complete! Check MLflow UI at http://localhost:5000")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()