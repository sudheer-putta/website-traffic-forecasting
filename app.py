import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import io
import os
from glob import glob

# Defensive import for statsmodels.ARIMA to avoid crash if package missing
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAVE_ARIMA = True
except Exception:
    ARIMA = None
    HAVE_ARIMA = False

st.set_page_config(layout="wide", page_title="Website Traffic Forecasting", initial_sidebar_state="expanded")

PROJECT_TITLE = "Website Traffic Forecasting — Daily Visits Forecasting"
st.title(PROJECT_TITLE)
st.markdown(
    "Develop robust forecasts of daily website traffic using both time-series (ARIMA) and "
    "machine-learning (Random Forest) approaches. Upload your CSV (must contain a 'Date' column) "
    "or use the default `daily-website-visitors.csv` file in the app folder."
)

# ---------------------
# Data loading & utils
# ---------------------
@st.cache_data
def find_csv_files(search_dirs=None):
    search_dirs = search_dirs or [
        ".",
        "./data",
        "./data/processed",
        "./website-traffic-forecasting/data/processed",
    ]
    files = []
    for d in search_dirs:
        try:
            files.extend(glob(os.path.join(d, "*.csv")))
        except Exception:
            continue
    # make unique and sort
    files = sorted(list(dict.fromkeys(files)))
    return files

@st.cache_data
def load_data(uploaded_file=None, selected_path=None):
    # Priority: uploaded_file > selected_path > default filename in ./data or cwd
    try:
        # If user uploaded via Streamlit widget, read that first
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file)

        # If user selected/entered a path in the sidebar, use it
        if selected_path:
            return pd.read_csv(selected_path)

        # Prefer CSV inside the project's data folder
        preferred = os.path.join("data", "daily-website-visitors.csv")
        if os.path.exists(preferred):
            return pd.read_csv(preferred)

        # Fallback to CSV in current working directory
        default_path = "daily-website-visitors.csv"
        if os.path.exists(default_path):
            return pd.read_csv(default_path)

        # Nothing found
        return None
    except Exception:
        return None

@st.cache_data
def preprocess(df):
    df = df.copy()
    if 'Date' not in df.columns:
        return pd.DataFrame()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values('Date').set_index('Date')
    df = df.fillna(method='ffill').fillna(method='bfill')

    # normalize column names
    df.columns = df.columns.str.strip().str.replace(' ', '.', regex=False)

    # safe numeric cleaning for expected columns
    numeric_cols = [
        'Page.Loads',
        'Unique.Visits',
        'First.Time.Visits',
        'Returning.Visits'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(',', '', regex=False)
                .replace({'nan': np.nan, 'None': np.nan})
            )
            # coerce to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # derive Day.Of.Week from index
    df['Day.Of.Week'] = df.index.dayofweek

    # fill remaining numerics
    for c in df.select_dtypes(include=['float', 'int']).columns:
        df[c] = df[c].fillna(method='ffill').fillna(method='bfill')

    return df

def create_features(data):
    df_feat = data.copy()
    df_feat['lag_1'] = df_feat['Page.Loads'].shift(1)
    df_feat['lag_7'] = df_feat['Page.Loads'].shift(7)
    df_feat['rolling_mean_7'] = df_feat['Page.Loads'].rolling(7, min_periods=1).mean()
    return df_feat.dropna()

def evaluate(actual, pred):
    if actual is None or pred is None:
        return np.nan, np.nan
    a, p = actual.align(pred, join='inner')
    if len(a) == 0:
        return np.nan, np.nan
    rmse = np.sqrt(mean_squared_error(a, p))
    try:
        mape = mean_absolute_percentage_error(a, p) * 100
    except Exception:
        mape = np.nan
    return rmse, mape

def forecast_to_csv(series):
    # return CSV as a string (Streamlit accepts str/bytes)
    return series.rename("forecast").to_csv()

# ---------------------
# Model training
# ---------------------
def train_arima(y_train, steps):
    # return Series indexed with forecast dates
    if y_train is None or len(y_train) == 0 or steps <= 0:
        return pd.Series(dtype=float)
    if not HAVE_ARIMA:
        last = float(y_train.iloc[-1])
        start = y_train.index[-1] + pd.Timedelta(days=1)
        idx = pd.date_range(start=start, periods=steps, freq='D')
        return pd.Series([last] * steps, index=idx)
    try:
        model = ARIMA(y_train, order=(2, 1, 2))
        fit = model.fit()
        fc = fit.forecast(steps=steps)
        start = y_train.index[-1] + pd.Timedelta(days=1)
        idx = pd.date_range(start=start, periods=steps, freq='D')
        return pd.Series(fc, index=idx)
    except Exception:
        last = float(y_train.iloc[-1])
        start = y_train.index[-1] + pd.Timedelta(days=1)
        idx = pd.date_range(start=start, periods=steps, freq='D')
        return pd.Series([last] * steps, index=idx)

def train_random_forest(X_train, y_train, param_grid):
    rf = RandomForestRegressor(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_

# ---------------------
# UI - Sidebar controls
# ---------------------
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# show CSV files found in common project locations
found_files = find_csv_files()
selected_path = None
if found_files:
    selected_path = st.sidebar.selectbox("Or select a CSV from project", ["(none)"] + found_files)
    if selected_path == "(none)":
        selected_path = None

# allow manual path if user prefers
manual_path = st.sidebar.text_input("Or enter full CSV path", value="")
if manual_path.strip() != "":
    selected_path = manual_path.strip()

train_frac = st.sidebar.slider("Train fraction (time series split)", 0.5, 0.9, 0.8, 0.05)
forecast_horizon = st.sidebar.number_input("Forecast horizon (days)", min_value=1, max_value=365, value=30, step=1)
model_choice = st.sidebar.multiselect("Models to run", ["ARIMA", "RandomForest"], default=["ARIMA", "RandomForest"])
rf_n_estimators = st.sidebar.selectbox("RF n_estimators", [50, 100, 200], index=1)
rf_max_depth = st.sidebar.selectbox("RF max_depth", [5, 10, 20, None], index=1)
run_button = st.sidebar.button("Run forecasting")

# ---------------------
# Load & preprocess (improved error reporting)
# ---------------------
raw = load_data(uploaded_file if uploaded_file is not None else None, selected_path)
if raw is None:
    st.error(
        "Unable to read CSV. Place `daily-website-visitors.csv` in the app folder, "
        "upload a file, or select one from the sidebar. Detected CSVs:\n\n" +
        ("\n".join(found_files) if found_files else "None found")
    )
    st.stop()

if st.checkbox("Show raw data (first 10 rows)"):
    st.dataframe(raw.head(10))

df = preprocess(raw)
if df.empty:
    st.error("Preprocessing failed or 'Date' column missing.")
    st.stop()

st.write("Processed data range:", f"{df.index.min().date()} to {df.index.max().date()}", f"({len(df)} rows)")

# Feature engineering
ml_data = create_features(df)
required_features = [
    'Day.Of.Week',
    'Unique.Visits',
    'First.Time.Visits',
    'Returning.Visits',
    'lag_1',
    'lag_7',
    'rolling_mean_7'
]
missing = [f for f in required_features if f not in ml_data.columns]
if missing:
    st.warning(f"Missing expected columns after preprocessing: {missing}. ML models may not run.")

# Train/test split
train_size = max(1, int(len(df) * train_frac))
train_end_date = df.index[train_size - 1]
train_ts, test_ts = df.iloc[:train_size], df.iloc[train_size:]
y_train_ts = train_ts['Page.Loads']
# prepare ML split aligned by date using ml_data
X = ml_data[required_features].copy() if all(f in ml_data.columns for f in required_features) else pd.DataFrame()
y_ml = ml_data['Page.Loads'] if 'Page.Loads' in ml_data.columns else pd.Series(dtype=float)
X_train = X.loc[:train_end_date] if not X.empty else pd.DataFrame()
X_test = X.loc[train_end_date + pd.Timedelta(days=1):] if not X.empty else pd.DataFrame()
y_train_ml = y_ml.loc[X_train.index] if not X.empty else pd.Series(dtype=float)
y_test_ml = y_ml.loc[X_test.index] if not X.empty else pd.Series(dtype=float)

# Wait for user to run
if not run_button:
    st.info("Configure settings in the sidebar and click 'Run forecasting'.")
    st.stop()

# Validate splits
if len(test_ts) == 0:
    st.error("No test period available. Reduce train fraction.")
    st.stop()
if ("RandomForest" in model_choice) and (X_train.empty or X_test.empty):
    st.error("Random Forest selected but ML features missing or splits empty.")
    st.stop()

# ---------------------
# Run models
# ---------------------
results = {}
with st.spinner("Running selected models..."):
    if "ARIMA" in model_choice:
        arima_forecast = train_arima(y_train_ts, steps=forecast_horizon)
        # try align to test_ts tail if lengths match
        if len(arima_forecast) == len(test_ts):
            try:
                arima_forecast.index = test_ts.index
            except Exception:
                pass
        results['ARIMA'] = arima_forecast

    if "RandomForest" in model_choice:
        param_grid = {'n_estimators': [rf_n_estimators], 'max_depth': [rf_max_depth]}
        best_rf, best_params = train_random_forest(X_train, y_train_ml, param_grid)
        rf_pred = best_rf.predict(X_test)
        rf_forecast = pd.Series(rf_pred, index=y_test_ml.index)
        results['RandomForest'] = {'model': best_rf, 'forecast': rf_forecast, 'params': best_params}

# ---------------------
# Evaluation & display
# ---------------------
st.header("Forecast Results & Evaluation")

cols = st.columns(len(results) if len(results) > 0 else 1)
i = 0
for name, res in results.items():
    col = cols[i]
    with col:
        st.subheader(name)
        if name == 'ARIMA':
            fc = res
            # if there is an overlapping actual test window, evaluate against test_ts head/tail
            # prefer overlap with test_ts if possible
            actual_for_eval = None
            if fc.index.equals(test_ts.index[:len(fc)]):
                actual_for_eval = test_ts['Page.Loads'].loc[fc.index]
            elif len(test_ts) >= len(fc):
                actual_for_eval = test_ts['Page.Loads'].iloc[:len(fc)]
                actual_for_eval.index = fc.index  # align for metric purpose only
            rmse, mape = evaluate(actual_for_eval, fc) if actual_for_eval is not None else (np.nan, np.nan)
            st.metric("RMSE", f"{rmse:.2f}" if not np.isnan(rmse) else "n/a")
            st.metric("MAPE", f"{mape:.2f}%" if not np.isnan(mape) else "n/a")
            st.write("Forecast (first 10 rows):")
            st.dataframe(fc.head(10))
            st.download_button(
                "Download ARIMA forecast CSV",
                data=forecast_to_csv(fc),
                file_name="arima_forecast.csv",
                mime="text/csv"
            )
        else:
            # RandomForest
            rf_info = res
            fc = rf_info['forecast']
            rmse, mape = evaluate(y_test_ml, fc)
            st.metric("RMSE", f"{rmse:.2f}" if not np.isnan(rmse) else "n/a")
            st.metric("MAPE", f"{mape:.2f}%" if not np.isnan(mape) else "n/a")
            st.write("Best RF params:")
            st.write(rf_info.get('params'))
            st.write("Feature importances:")
            fi = pd.Series(rf_info['model'].feature_importances_, index=X_train.columns).sort_values(ascending=False)
            st.dataframe(fi.reset_index().rename(columns={'index': 'feature', 0: 'importance'}))
            st.write("Forecast (first 10 rows):")
            st.dataframe(fc.head(10))
            st.download_button(
                "Download RF forecast CSV",
                data=forecast_to_csv(fc),
                file_name="rf_forecast.csv",
                mime="text/csv"
            )
    i += 1

# ---------------------
# Combined plots
# ---------------------
st.subheader("Comparison plot")
plot_items = {}
# Plot actual recent Page.Loads
plot_items["Actual"] = df['Page.Loads'].iloc[-(forecast_horizon*2):]  # show recent window
for name, res in results.items():
    if name == 'ARIMA':
        plot_items["ARIMA Forecast"] = res
    else:
        plot_items["RandomForest Forecast"] = res['forecast']
# Build DataFrame for plotting (align on index)
plot_df = pd.concat(plot_items, axis=1)
st.line_chart(plot_df)

st.markdown("Project: Forecast daily website visits to help content planning and resource allocation.")