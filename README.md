# MLflow Feature Analysis Dashboard

A production-ready web application that extends MLflow with interactive SHAP-based feature analysis and real-time model explainability.

## 🎯 Project Overview

This project creates a **Feature Analysis Tab** for MLflow that allows you to:
- 📊 Upload datasets (Titanic, Iris, Hotel Booking)
- 🤖 Select trained models from MLflow
- ⚡ Compute SHAP values **in real-time** with live progress updates
- 📈 View interactive feature importance visualizations
- 💾 Download and compare SHAP results

### Tech Stack
- **Backend**: FastAPI + Python 3.10+
- **Frontend**: React 18 + TypeScript
- **Visualization**: Plotly.js
- **Real-time**: WebSocket (live progress)
- **ML Tools**: MLflow, SHAP, scikit-learn, pandas
- **Deployment**: Docker Compose

---

## 📁 Project Structure

```
mlflow-feature-analysis/
│
├── backend/
│   ├── main.py                          # FastAPI app entry point
│   ├── requirements.txt                 # Python dependencies
│   ├── Dockerfile
│   │
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py                    # Configuration (MLflow URI, etc)
│   │   ├── models.py                    # Pydantic schemas
│   │   │
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── runs.py                  # /api/runs endpoints
│   │   │   ├── shap.py                  # /api/shap endpoints + WebSocket
│   │   │   └── upload.py                # /api/upload endpoints
│   │   │
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── mlflow_service.py        # MLflow interactions
│   │       ├── shap_service.py          # SHAP computation
│   │       └── file_service.py          # CSV handling
│   │
│   ├── notebooks/
│   │   ├── 01_train_titanic.ipynb       # Titanic model training
│   │   ├── 02_train_iris.ipynb          # Iris model training
│   │   └── 03_train_hotel.ipynb         # Hotel booking model training
│   │
│   └── data/
│       ├── titanic.csv
│       ├── iris.csv
│       └── hotel_booking.csv
│
├── frontend/
│   ├── package.json
│   ├── vite.config.ts                   # Vite configuration
│   ├── Dockerfile
│   │
│   ├── public/
│   │   └── index.html
│   │
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       │
│       ├── components/
│       │   ├── FeatureAnalysis.tsx       # Main dashboard component
│       │   ├── RunSelector.tsx           # MLflow run dropdown
│       │   ├── DataUpload.tsx            # CSV upload form
│       │   ├── ShapVisualizer.tsx        # SHAP plots display
│       │   ├── ProgressTracker.tsx       # Real-time progress
│       │   └── ExportResults.tsx         # Download SHAP results
│       │
│       ├── services/
│       │   └── api.ts                    # FastAPI client
│       │
│       ├── types/
│       │   └── index.ts                  # TypeScript interfaces
│       │
│       └── styles/
│           └── App.css
│
├── docker-compose.yml                   # Run all services
├── .gitignore
├── .env.example                         # Environment variables template
│
├── docs/
│   ├── SETUP.md                         # Installation guide
│   ├── API.md                           # Backend API documentation
│   ├── ARCHITECTURE.md                  # System design
│   └── USAGE.md                         # How to use the dashboard
│
└── README.md                            # Main project readme
```

---

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Docker & Docker Compose
- Python 3.10+
- Node.js 18+
- MLflow (will be installed via pip)

### Option 1: Using Docker (Recommended)

```bash
# Clone the repo
git clone https://github.com/YOUR-USERNAME/mlflow-feature-analysis.git
cd mlflow-feature-analysis

# Start all services (MLflow, FastAPI, React)
docker-compose up

# Access:
# - MLflow UI: http://localhost:5000
# - Dashboard: http://localhost:3000
# - Backend API: http://localhost:8000
```

### Option 2: Local Setup

```bash
# Terminal 1 - Start MLflow server
mlflow server --host 0.0.0.0 --port 5000

# Terminal 2 - Start FastAPI backend
cd backend
pip install -r requirements.txt
python main.py
# Runs on http://localhost:8000

# Terminal 3 - Start React frontend
cd frontend
npm install
npm run dev
# Runs on http://localhost:3000
```

---

## 📊 Workflow

1. **Start MLflow** (`http://localhost:5000`)
2. **Run training scripts** (Jupyter notebooks in `backend/notebooks/`)
   - This logs models to MLflow
   - Models appear in MLflow UI
3. **Open Dashboard** (`http://localhost:3000`)
4. **Upload Dataset** (Titanic/Iris/Hotel CSV)
5. **Select MLflow Run** (trained model)
6. **Click "Compute SHAP"**
   - See real-time progress via WebSocket
   - Visualize SHAP values interactively
7. **Export Results** (download JSON/CSV)

---

## 🔧 Configuration

### Backend (`backend/.env`)
```bash
MLFLOW_TRACKING_URI=http://localhost:5000
UPLOAD_DIR=./uploads
MAX_FILE_SIZE=10485760  # 10MB
```

### Frontend (`frontend/.env`)
```bash
VITE_API_URL=http://localhost:8000
VITE_API_WS_URL=ws://localhost:8000
```

---

## 📚 API Endpoints

### Runs Management
```
GET  /api/runs                      # List all MLflow runs
GET  /api/runs/{run_id}             # Get run details
GET  /api/runs/search?experiment=*  # Search runs
```

### Data Upload
```
POST /api/upload                    # Upload CSV file
  Response: {columns, shape, preview}
```

### SHAP Computation (Real-time)
```
POST /api/shap/compute              # Start SHAP computation
  Body: {run_id, model_type}

WebSocket /ws/shap/{computation_id} # Real-time progress updates
  Message: {status, progress, error}

GET  /api/shap/results/{computation_id}  # Fetch results
  Response: {shap_values, features, importance, plots}

GET  /api/shap/download/{computation_id} # Download as JSON
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER BROWSER (Port 3000)                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           React Dashboard (TypeScript)               │   │
│  │  ├─ Upload Form                                      │   │
│  │  ├─ Run Selector (queries MLflow)                    │   │
│  │  ├─ Plotly SHAP Visualizations                       │   │
│  │  └─ Real-time Progress Tracker (WebSocket)           │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                         │
                         │ HTTP / WebSocket
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI Backend (Port 8000)                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  /api/runs        → Queries MLflow server            │   │
│  │  /api/upload      → Validates & stores CSV files     │   │
│  │  /api/shap/*      → Computes SHAP values             │   │
│  │  /ws/shap/*       → Sends real-time progress updates │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                         │
                         │ REST API
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 MLflow Server (Port 5000)                   │
│  ├─ Experiments Tab                                        │
│  ├─ Models Registry                                        │
│  └─ REST API endpoints                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎓 Development Roadmap

### Phase 1 (Days 1-5): Foundation
- ✅ Setup project structure
- ✅ Create training scripts (Titanic, Iris)
- ✅ Build FastAPI backend with file upload
- ✅ Setup MLflow integration

### Phase 2 (Days 6-10): SHAP Integration
- ✅ Implement SHAP computation service
- ✅ Add WebSocket for real-time progress
- ✅ Create result caching system
- ✅ Build export functionality

### Phase 3 (Days 11-15): Frontend
- ✅ Setup React + TypeScript
- ✅ Create component hierarchy
- ✅ Integrate Plotly for visualizations
- ✅ Connect to FastAPI backend
- ✅ Testing & optimization

---

## 🧪 Testing

```bash
# Backend tests
cd backend
pytest tests/

# Frontend tests
cd frontend
npm test

# Integration test
bash scripts/test_integration.sh
```

---

## 📖 Documentation

- **[SETUP.md](docs/SETUP.md)** - Detailed installation & troubleshooting
- **[API.md](docs/API.md)** - Complete API reference
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System design & decisions
- **[USAGE.md](docs/USAGE.md)** - Step-by-step usage guide

---

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

---

## ✨ Key Features

✅ **Real-time SHAP Computation** - On-demand calculations with live progress
✅ **Multi-dataset Support** - Titanic, Iris, Hotel Booking (extensible)
✅ **Interactive Visualizations** - Zoom, hover, filter SHAP plots
✅ **Model Comparison** - Compare SHAP across different runs
✅ **Export Results** - Download SHAP values as JSON/CSV
✅ **Production-Ready** - Error handling, logging, validation
✅ **Fully Dockerized** - One command to run everything
✅ **TypeScript Frontend** - Type-safe React components
✅ **Professional UI** - Modern, responsive design

---

## 🐛 Troubleshooting

### MLflow not connecting
```bash
# Check if MLflow is running
curl http://localhost:5000/health

# If not, start it:
mlflow server --host 0.0.0.0 --port 5000
```

### SHAP computation is slow
- Increase dataset size gradually
- Use tree-based models (faster SHAP)
- Check backend logs: `docker logs mlflow-backend`

### React not connecting to backend
- Check if FastAPI is running: `curl http://localhost:8000/docs`
- Verify CORS is enabled in `backend/main.py`
- Check browser console for errors

---

## 📞 Support

For issues, questions, or suggestions:
1. Check [docs/](docs/) folder
2. Open a GitHub Issue
3. See [Discussions](https://github.com/YOUR-USERNAME/mlflow-feature-analysis/discussions)

---

## 🎉 What's Next?

After Phase 1 completes:
- Add fairness evaluation metrics
- Implement model comparison dashboard
- Create experiment tracking timeline
- Add batch SHAP computation
- Deploy to cloud (AWS/GCP/Azure)
