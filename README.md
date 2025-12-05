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

### Local Development

```bash
# Clone the repository
git clone <repo-url>
cd junon-time-series

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -e .

# Run the app
streamlit run dashboard/training/app.py
```

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
│   │   ├── app.py          # Main entry point
│   │   └── pages/
│   │       ├── 1_Train_Models.py
│   │       └── 2_Forecasting.py
│   ├── explorer/           # Data exploration app (optional)
│   ├── utils/              # Shared utilities
│   │   ├── model_factory.py
│   │   ├── preprocessing.py
│   │   ├── timeshap_wrapper.py
│   │   └── ...
│   └── models_config.py    # Model definitions
├── checkpoints/            # Saved models
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
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

---

## License

MIT License - See [LICENSE](LICENSE)

## Author

Nicolas Ringuet - Université de Tours
