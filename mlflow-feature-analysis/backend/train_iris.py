# backend/train_iris.py
"""
Training script for Iris dataset
Logs model to MLflow for use in Feature Analysis dashboard
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MLFLOW_TRACKING_URI = "http://localhost:5000"
DATA_PATH = "./data/iris.csv"  # Replace with your path

def load_and_preprocess_data(filepath):
    """Load and preprocess Iris dataset"""
    logger.info("Loading Iris dataset...")
    
    df = pd.read_csv(filepath)
    
    # Assuming last column is target
    # If columns are different, adjust accordingly
    if 'species' in df.columns:
        target_col = 'species'
    elif 'target' in df.columns:
        target_col = 'target'
    elif df.columns[-1] in ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']:
        # Column name might be the species name
        # Find the column with species
        for col in df.columns:
            if 'iris' in col.lower() or 'setosa' in str(df[col].unique()).lower():
                target_col = col
                break
    else:
        target_col = df.columns[-1]
    
    # Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Encode target if categorical
    if y.dtype == 'object':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Features: {X.columns.tolist()}")
    logger.info(f"Classes: {np.unique(y)}")
    
    return X, y

def train_and_log_model(X, y, model_type='dt', run_name='iris_baseline'):
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
    if model_type == 'dt':
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        model_name = "decision_tree"
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model_name = "random_forest"
    
    # Train
    logger.info(f"Training {model_name}...")
    model.fit(X_train_scaled, y_train)
    
    # Evaluate (multiclass)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
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
        elif model_type == 'dt':
            mlflow.log_param("max_depth", 5)
        
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
        
        # Train Decision Tree
        train_and_log_model(X, y, model_type='dt', run_name='iris_decision_tree')
        
        # Train Random Forest
        train_and_log_model(X, y, model_type='rf', run_name='iris_random_forest')
        
        logger.info("Training complete! Check MLflow UI at http://localhost:5000")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()

