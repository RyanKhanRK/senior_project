# backend/main.py
from fastapi import FastAPI, UploadFile, File, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import mlflow
import pandas as pd
import numpy as np
import shap
import io
import asyncio
import uuid
from typing import Dict, Optional
import logging
from datetime import datetime
import joblib
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="MLflow Feature Analysis API",
    description="Real-time SHAP computation for MLflow models",
    version="1.0.0"
)

# CORS middleware - allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MLFLOW_TRACKING_URI = "http://localhost:5000"
UPLOAD_DIR = "./uploads"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# In-memory storage for computation results and progress
computation_results: Dict[str, dict] = {}
computation_progress: Dict[str, dict] = {}

# ==================== MODELS ====================

class RunResponse:
    def __init__(self, run_id: str, name: str, status: str, created_at: str):
        self.run_id = run_id
        self.name = name
        self.status = status
        self.created_at = created_at

# ==================== ROUTES: RUNS ====================

@app.get("/api/runs")
async def get_runs():
    """
    Fetch all MLflow experiment runs
    """
    try:
        runs = mlflow.search_runs()
        if runs.empty:
            return {"runs": [], "total": 0}
        
        result = []
        for _, row in runs.iterrows():
            result.append({
                "run_id": row["run_id"],
                "experiment_id": row["experiment_id"],
                "status": row["status"],
                "start_time": str(row["start_time"]),
                "end_time": str(row["end_time"]) if pd.notna(row["end_time"]) else None,
                "artifact_uri": row["artifact_uri"]
            })
        
        return {
            "runs": result,
            "total": len(result)
        }
    except Exception as e:
        logger.error(f"Error fetching runs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/runs/{run_id}")
async def get_run_details(run_id: str):
    """
    Get details for a specific run including params, metrics, artifacts
    """
    try:
        run = mlflow.get_run(run_id)
        return {
            "run_id": run.info.run_id,
            "status": run.info.status,
            "params": dict(run.data.params),
            "metrics": dict(run.data.metrics),
            "tags": dict(run.data.tags),
        }
    except Exception as e:
        logger.error(f"Error fetching run details: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

# ==================== ROUTES: UPLOAD ====================

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and validate CSV file
    """
    try:
        # Check file size
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
        
        # Read CSV
        df = pd.read_csv(io.BytesIO(content))
        
        return {
            "filename": file.filename,
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "preview": df.head(5).to_dict(orient="records"),
            "missing_values": df.isnull().sum().to_dict()
        }
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {str(e)}")
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ROUTES: SHAP ====================

@app.post("/api/shap/compute")
async def compute_shap(run_id: str, file: UploadFile = File(...)):
    """
    Start SHAP computation for a model run
    Returns computation_id to track progress via WebSocket
    """
    try:
        # Validate inputs
        if not run_id:
            raise HTTPException(status_code=400, detail="run_id required")
        
        # Generate unique computation ID
        computation_id = str(uuid.uuid4())
        
        # Initialize progress tracking
        computation_progress[computation_id] = {
            "status": "initializing",
            "progress": 0,
            "error": None,
            "start_time": datetime.now().isoformat()
        }
        
        # Read CSV data
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # Start async computation
        asyncio.create_task(
            _compute_shap_async(computation_id, run_id, df)
        )
        
        return {
            "computation_id": computation_id,
            "status": "queued"
        }
    except Exception as e:
        logger.error(f"Error starting SHAP computation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def _compute_shap_async(computation_id: str, run_id: str, df: pd.DataFrame):
    """
    Background task: Load model, compute SHAP values
    """
    try:
        # Step 1: Load model from MLflow (20% progress)
        computation_progress[computation_id]["status"] = "Loading model..."
        computation_progress[computation_id]["progress"] = 20
        
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        
        # Step 2: Prepare data (40% progress)
        computation_progress[computation_id]["status"] = "Preparing data..."
        computation_progress[computation_id]["progress"] = 40
        
        # Handle categorical columns
        X = df.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X = X[numeric_cols]
        
        # Step 3: Compute SHAP (70% progress)
        computation_progress[computation_id]["status"] = "Computing SHAP values..."
        computation_progress[computation_id]["progress"] = 70
        
        # Determine explainer type based on model
        if hasattr(model, 'tree_'):  # Tree-based model
            explainer = shap.TreeExplainer(model)
        else:
            # Use KernelExplainer for other models (slower but works with any model)
            explainer = shap.KernelExplainer(model.predict, X.iloc[:50])  # Use sample for speed
        
        shap_values = explainer.shap_values(X)
        
        # Handle multiclass output
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values).mean(axis=0)
        
        # Step 4: Calculate importance metrics (90% progress)
        computation_progress[computation_id]["status"] = "Calculating importance..."
        computation_progress[computation_id]["progress"] = 90
        
        feature_importance = np.abs(shap_values).mean(axis=0).tolist()
        feature_names = X.columns.tolist()
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": feature_importance
        }).sort_values("importance", ascending=False)
        
        # Step 5: Store results (100% progress)
        computation_results[computation_id] = {
            "shap_values": shap_values.tolist(),
            "features": feature_names,
            "feature_importance": importance_df.to_dict(orient="records"),
            "model_id": run_id,
            "dataset_shape": X.shape,
            "computed_at": datetime.now().isoformat()
        }
        
        computation_progress[computation_id]["status"] = "Complete"
        computation_progress[computation_id]["progress"] = 100
        
        logger.info(f"SHAP computation {computation_id} completed")
        
    except Exception as e:
        logger.error(f"SHAP computation error: {str(e)}")
        computation_progress[computation_id]["error"] = str(e)
        computation_progress[computation_id]["status"] = "Error"

@app.websocket("/ws/shap/{computation_id}")
async def websocket_shap_progress(websocket: WebSocket, computation_id: str):
    """
    WebSocket endpoint for real-time progress updates
    """
    await websocket.accept()
    try:
        while True:
            if computation_id in computation_progress:
                progress = computation_progress[computation_id]
                await websocket.send_json(progress)
                
                # If computation is done or errored, close connection
                if progress["status"] in ["Complete", "Error"]:
                    await asyncio.sleep(0.5)
                    break
            
            await asyncio.sleep(0.5)  # Update every 500ms
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()

@app.get("/api/shap/results/{computation_id}")
async def get_shap_results(computation_id: str):
    """
    Retrieve SHAP computation results
    """
    if computation_id not in computation_results:
        raise HTTPException(status_code=404, detail="Computation not found")
    
    return computation_results[computation_id]

@app.get("/api/shap/download/{computation_id}")
async def download_shap_results(computation_id: str):
    """
    Download SHAP results as JSON
    """
    if computation_id not in computation_results:
        raise HTTPException(status_code=404, detail="Computation not found")
    
    results = computation_results[computation_id]
    return JSONResponse(content=results)

# ==================== HEALTH CHECK ====================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "mlflow_uri": MLFLOW_TRACKING_URI,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "MLflow Feature Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "runs": "/api/runs",
            "upload": "/api/upload",
            "shap_compute": "/api/shap/compute",
            "shap_results": "/api/shap/results/{computation_id}",
            "health": "/health"
        }
    }

# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )