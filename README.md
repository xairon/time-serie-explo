# Time Series Forecasting Dashboard

A Streamlit-based dashboard for training and evaluating deep learning models on time series data, with a focus on hydrological forecasting (piezometric levels, groundwater).

## Features

### 🎯 Model Training
- **12+ Deep Learning models**: TFT, Transformer, N-BEATS, N-HiTS, LSTM, GRU, TCN, TiDE, TSMixer, DLinear, NLinear...
- **Automatic hyperparameter optimization** with Optuna
- **Real-time training visualization** (loss curves, progress)
- **Early stopping** to prevent overfitting
- **Covariate support**: temperature, precipitation, evapotranspiration...

### 🔮 Forecasting & Evaluation
- **Sliding window predictions** on test set
- **One-step and autoregressive** forecasting modes
- **Performance metrics**: MAE, RMSE, R², MAPE
- **Interactive charts** with Plotly

### 💡 Explainability (TimeSHAP)
- **Temporal importance**: Which past days influenced the prediction?
- **Feature importance**: Which variables contributed most?
- **Local explanations** for each prediction window

### 📊 Data Preprocessing
- Multiple normalization options (StandardScaler, MinMax)
- Missing value handling (interpolation, forward/backward fill)
- Automatic datetime features (day of week, month, cyclical encoding)
- Lag feature generation

---

## Quick Start

> 💡 **Pour un guide détaillé de déploiement**, consultez [DEPLOYMENT.md](DEPLOYMENT.md)

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd junon-time-series
   ```

2. **Setup Environment** (Automated)
   
   Run the setup script to create a virtual environment and install dependencies optimized for your hardware (CPU, NVIDIA CUDA, or Intel Arc XPU).

   ```bash
   # Interactive mode (recommended)
   python setup_env.py

   # Or specify target directly:
   python setup_env.py --device cpu   # For standard CPU
   python setup_env.py --device cuda  # For NVIDIA GPUs
   python setup_env.py --device xpu   # For Intel Arc GPUs
   
   # With custom venv name:
   python setup_env.py --device xpu --venv venv_arc
   ```

   The script will:
   - Create a virtual environment
   - Install PyTorch with the appropriate index-url for your device
   - Install all base dependencies
   - Verify the installation

3. **Verify Installation** (Optional)
   
   ```bash
   python verify_installation.py --venv venv --device cpu
   ```

4. **Activate Environment**
   
   - Windows: `venv\Scripts\activate` (or `venv_arc\Scripts\activate` for XPU)
   - Linux/Mac: `source venv/bin/activate`

5. **Run the app**
   
   ```bash
   # Using the run script (recommended)
   python run_app.py
   
   # Or directly with Streamlit
   streamlit run dashboard/training/Home.py
   
   # With custom port
   python run_app.py --port 8502
   ```
   
   The app will be available at `http://localhost:8501` (or your custom port).

### Docker Deployment

```bash
# Build and run
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

The app runs on port **49500** (Nicolas block: 49500-49599).

Access: `http://your-server:49500`

---

## How It Works

### 1. Upload Data
Upload a CSV file with:
- A **datetime column** (index)
- A **target variable** (what you want to predict)
- Optional **covariates** (explanatory variables)

Example format:
```csv
date,water_level,temperature,precipitation
2020-01-01,12.5,8.2,0.0
2020-01-02,12.3,9.1,2.5
...
```

### 2. Configure Preprocessing
- Select target and covariate columns
- Choose normalization method
- Handle missing values
- Add datetime features or lags

### 3. Select & Train Model
- Choose a model architecture (TFT recommended for complex series)
- Configure hyperparameters manually or use Optuna
- Set train/validation/test splits
- Monitor training in real-time

### 4. Evaluate & Explain
- View predictions vs ground truth
- Analyze errors and residuals
- Use TimeSHAP to understand model decisions
- Export predictions as CSV

---

## Project Structure

```
├── dashboard/
│   ├── training/           # Training app
│   │   ├── Home.py        # Main entry point
│   │   └── pages/
│   │       ├── 1_Dataset_Preparation.py
│   │       ├── 2_Train_Models.py
│   │       └── 3_Forecasting.py
│   ├── explorer/           # Data exploration app (optional)
│   ├── utils/              # Shared utilities
│   │   ├── model_factory.py
│   │   ├── preprocessing.py
│   │   ├── timeshap_wrapper.py
│   │   └── ...
│   └── config.py           # Configuration
├── requirements/           # Dependency files
│   ├── base.txt           # Base dependencies
│   ├── cpu.txt            # CPU-specific (PyTorch CPU)
│   ├── cuda.txt           # CUDA-specific (PyTorch CUDA)
│   └── xpu.txt            # XPU-specific (PyTorch XPU)
├── setup_env.py           # Automated environment setup
├── verify_installation.py # Installation verification
├── run_app.py             # Streamlit launcher
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## Configuration

### Port

The application runs on port **49500** (fixed, from Nicolas block 49500-49599).

### Streamlit Config

Edit `.streamlit/config.toml` to customize theme, server settings, etc.

---

## Requirements

- Python 3.10 - 3.12
- PyTorch 2.0+ (CPU, CUDA 11.8, or Intel XPU)
- Virtual environment (created automatically by `setup_env.py`)

### Architecture-Specific Requirements

- **CPU**: No additional requirements
- **CUDA**: NVIDIA GPU with CUDA 11.8 support
- **XPU**: Intel Arc GPU with Intel oneAPI Base Toolkit

## Troubleshooting

### Installation Issues

If you encounter issues during installation:

1. **Verify your Python version**:
   ```bash
   python --version  # Should be 3.10, 3.11, or 3.12
   ```

2. **Check virtual environment**:
   ```bash
   python verify_installation.py --venv venv
   ```

3. **Recreate environment**:
   ```bash
   # Remove old venv
   rm -rf venv  # Linux/Mac
   rmdir /s venv  # Windows
   
   # Recreate
   python setup_env.py --device cpu
   ```

### PyTorch Installation Issues

- **CPU**: Should work automatically
- **CUDA**: Ensure you have NVIDIA drivers and CUDA 11.8 installed
- **XPU**: Requires Intel Arc GPU and Intel oneAPI Base Toolkit

### Streamlit Issues

If Streamlit fails to start:
- Check that the virtual environment is activated
- Verify installation: `python verify_installation.py`
- Try running directly: `streamlit run dashboard/training/Home.py`

---

## License

MIT License - See [LICENSE](LICENSE)

## Author

Nicolas Ringuet - Université de Tours
