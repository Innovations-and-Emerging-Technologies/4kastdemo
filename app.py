import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import logging
from statsmodels.tsa.api import VAR, VARMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
from pmdarima import auto_arima
from io import StringIO

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Function to load CSV file
def load_csv(file):
    try:
        df = pd.read_csv(file)
        logging.info("File loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading file: {e}")
        return None

def clean_demand_column(df):
    try:
        # Interpolate missing values in the 'Demand' column
        df['Demand'] = pd.to_numeric(df['Demand'], errors='coerce')
        df['Demand'] = df['Demand'].interpolate(method='linear')
        
        # Interpolate missing values in additional columns
        for col in df.columns:
            if col.endswith('_mapped'):
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].interpolate(method='linear')
        
        return df
    except Exception as e:
        logging.error(f"Error cleaning columns: {e}")
        return None

# Function to normalize date formats
def normalize_dates(df, date_col):
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        if df[date_col].isnull().any():
            raise ValueError("Some dates could not be parsed. Please check the date format.")
        df = df.sort_values(by=date_col)
        logging.info("Dates successfully normalized and sorted.")
        return df
    except Exception as e:
        logging.error(f"Error normalizing dates: {e}")
        return None

# Function to map columns
def map_columns(df, date_col, demand_col, additional_cols=None):
    try:
        df.rename(columns={date_col: 'Date', demand_col: 'Demand'}, inplace=True)
        if additional_cols:
            for col in additional_cols:
                df.rename(columns={col: col + '_mapped'}, inplace=True)
        df = normalize_dates(df, 'Date')
        df = clean_demand_column(df)
        logging.info("Columns successfully mapped!")
        return df
    except Exception as e:
        logging.error(f"Error mapping columns: {e}")
        return None

# Function to infer frequency
def infer_frequency(df, date_col='Date'):
    """
    Infer the frequency of the time series data.
    """
    try:
        freq = pd.infer_freq(df[date_col])
        if freq is None:
            logging.warning("Unable to infer frequency. Defaulting to daily ('D').")
            freq = 'D'
        return freq
    except Exception as e:
        logging.error(f"Error inferring frequency: {e}")
        return 'D'  # Default to daily frequency

# Function for EDA
def perform_eda(df):
    st.subheader("Exploratory Data Analysis (EDA)")
    
    # Display basic statistics
    st.write("### Data Summary")
    st.write(df.describe())
    
    # Check for missing values
    st.write("### Missing Values")
    st.write(df.isnull().sum())
    
    # Plot demand over time
    st.write("### Demand Trend")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['Date'], df['Demand'], label='Historical Demand')
    ax.set_xlabel('Date')
    ax.set_ylabel('Demand')
    ax.set_title('Demand Trend')
    ax.legend()
    st.pyplot(fig)
    
    # Seasonality and trend decomposition
    st.write("### Seasonality and Trend Decomposition")
    decomposition = seasonal_decompose(df.set_index('Date')['Demand'], model='additive', period=30)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    decomposition.trend.plot(ax=ax1, title='Trend')
    decomposition.seasonal.plot(ax=ax2, title='Seasonality')
    decomposition.resid.plot(ax=ax3, title='Residuals')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Autocorrelation and Partial Autocorrelation
    st.write("### Autocorrelation and Partial Autocorrelation")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(df['Demand'], lags=30, ax=ax1)
    plot_pacf(df['Demand'], lags=30, ax=ax2)
    plt.tight_layout()
    st.pyplot(fig)

# Function to detect seasonality
def detect_seasonality(df, max_lags=50):
    st.write("### Seasonality Detection")
    # Plot ACF and PACF
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(df['Demand'], lags=max_lags, ax=ax1)
    plot_pacf(df['Demand'], lags=max_lags, ax=ax2)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Seasonal decomposition
    decomposition = seasonal_decompose(df.set_index('Date')['Demand'], model='additive', period=30)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    decomposition.trend.plot(ax=ax1, title='Trend')
    decomposition.seasonal.plot(ax=ax2, title='Seasonality')
    decomposition.resid.plot(ax=ax3, title='Residuals')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Infer seasonal period
    seasonal_period = infer_seasonal_period(df)
    st.write(f"Detected seasonal period: {seasonal_period}")
    return seasonal_period

# Function to infer seasonal period
def infer_seasonal_period(df):
    freq = pd.infer_freq(df['Date'])
    if freq == 'M':  # Monthly data
        return 12
    elif freq == 'W':  # Weekly data
        return 52
    elif freq == 'D':  # Daily data
        return 7
    else:
        logging.warning("Unable to infer seasonal period. Defaulting to 12.")
        return 12

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        logging.warning("Invalid predictions: NaN or Inf values detected.")
        return None, None, None
    try:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        return rmse, mape, mae
    except Exception as e:
        logging.error(f"Error calculating metrics: {e}")
        return None, None, None

# Function to auto-tune SARIMA
def auto_tune_sarima(train, seasonal_period):
    model = auto_arima(
        train['Demand'],
        seasonal=True,
        m=seasonal_period,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore"
    )
    return model


# Function to forecast using various models
def forecast_models(df, selected_models, additional_cols=None, item_col=None):
    # Create a copy of the dataframe to preserve the original
    df_copy = df.copy()
    
    if additional_cols is None:
        additional_cols = []
    
    # Split the data into train, validation, and test sets
    train_val, test = train_test_split(df_copy, test_size=0.2, shuffle=False)
    train, val = train_test_split(train_val, test_size=0.25, shuffle=False)
    
    # Store the dates before setting index
    dates = df_copy['Date'].copy()
    
    # Set index for each dataset
    train.set_index('Date', inplace=True)
    val.set_index('Date', inplace=True)
    test.set_index('Date', inplace=True)
    
    results = {}
    future_forecasts = {}
    validation_predictions = {}

    # Ensure 'Demand' column is of type float64
    train['Demand'] = train['Demand'].astype('float64')
    val['Demand'] = val['Demand'].astype('float64')
    test['Demand'] = test['Demand'].astype('float64')

    # Automatically detect seasonality using training data only
    seasonal_period = detect_seasonality(train.reset_index())

    # AR Model
    if 'AR' in selected_models:
        try:
            model = AutoReg(train['Demand'], lags=2).fit()
            
            # Training performance
            train_forecast = model.predict(start=0, end=len(train)-1)
            train_rmse, train_mape, train_mae = calculate_metrics(train['Demand'].values, train_forecast)
            
            # Validation performance
            val_forecast = model.predict(start=len(train), end=len(train)+len(val)-1)
            val_rmse, val_mape, val_mae = calculate_metrics(val['Demand'].values, val_forecast[-len(val):])
            
            validation_predictions['AR'] = {
                'actual': val['Demand'].values,
                'predicted': val_forecast[-len(val):],
                'dates': val.index
            }
            
            # Test performance
            test_forecast = model.predict(start=len(train)+len(val), end=len(train)+len(val)+len(test)-1)
            test_rmse, test_mape, test_mae = calculate_metrics(test['Demand'].values, test_forecast[-len(test):])
            
            results['AR'] = {
                'train': (train_rmse, train_mape, train_mae),
                'val': (val_rmse, val_mape, val_mae),
                'test': (test_rmse, test_mape, test_mae)
            }
            
            # Future forecast
            future_forecast = model.predict(start=len(df_copy), end=len(df_copy)+29)
            future_forecasts['AR'] = future_forecast.tolist()
        except Exception as e:
            results['AR'] = str(e)

    # ARMA Model
    if 'ARMA' in selected_models:
        try:
            model = ARIMA(train['Demand'], order=(2, 0, 1)).fit()
            
            # Training performance
            train_forecast = model.predict(start=0, end=len(train)-1)
            train_rmse, train_mape, train_mae = calculate_metrics(train['Demand'].values, train_forecast)
            
            # Validation performance
            val_forecast = model.predict(start=len(train), end=len(train)+len(val)-1)
            val_rmse, val_mape, val_mae = calculate_metrics(val['Demand'].values, val_forecast)
            
            validation_predictions['ARMA'] = {
                'actual': val['Demand'].values,
                'predicted': val_forecast,
                'dates': val.index
            }
            
            # Test performance
            test_forecast = model.predict(start=len(train)+len(val), end=len(train)+len(val)+len(test)-1)
            test_rmse, test_mape, test_mae = calculate_metrics(test['Demand'].values, test_forecast)
            
            results['ARMA'] = {
                'train': (train_rmse, train_mape, train_mae),
                'val': (val_rmse, val_mape, val_mae),
                'test': (test_rmse, test_mape, test_mae)
            }
            
            # Future forecast
            future_forecast = model.predict(start=len(df_copy), end=len(df_copy)+29)
            future_forecasts['ARMA'] = future_forecast.tolist()
        except Exception as e:
            results['ARMA'] = str(e)

    # SARIMA Model
    if 'SARIMA' in selected_models:
        try:
            model = auto_tune_sarima(train, seasonal_period)
            
            # Training performance
            train_forecast = model.predict_in_sample()
            train_rmse, train_mape, train_mae = calculate_metrics(train['Demand'].values, train_forecast)
            
            # Validation performance
            val_forecast = model.predict(n_periods=len(val))
            val_rmse, val_mape, val_mae = calculate_metrics(val['Demand'].values, val_forecast)
            
            validation_predictions['SARIMA'] = {
                'actual': val['Demand'].values,
                'predicted': val_forecast,
                'dates': val.index
            }
            
            # Test performance
            test_forecast = model.predict(n_periods=len(test))
            test_rmse, test_mape, test_mae = calculate_metrics(test['Demand'].values, test_forecast)
            
            results['SARIMA'] = {
                'train': (train_rmse, train_mape, train_mae),
                'val': (val_rmse, val_mape, val_mae),
                'test': (test_rmse, test_mape, test_mae)
            }
            
            # Future forecast
            future_forecast = model.predict(n_periods=30)
            future_forecasts['SARIMA'] = future_forecast.tolist()
        except Exception as e:
            results['SARIMA'] = str(e)

    if 'VAR' in selected_models and len(additional_cols) > 0:
        try:
            train_vars = train[['Demand'] + [col + '_mapped' for col in additional_cols]]
            model = VAR(train_vars)
            model_fitted = model.fit()
            
            # Training performance
            train_forecast = model_fitted.forecast(train_vars.values[-model_fitted.k_ar:], steps=len(train))
            train_rmse, train_mape, train_mae = calculate_metrics(train['Demand'].values, train_forecast[:, 0])

            # Validation performance
            val_forecast = model_fitted.forecast(train_vars.values[-model_fitted.k_ar:], steps=len(val))
            val_rmse, val_mape, val_mae = calculate_metrics(val['Demand'].values, val_forecast[:, 0])
            
            validation_predictions['VAR'] = {
                'actual': val['Demand'].values,
                'predicted': val_forecast[:, 0],
                'dates': val.index
            }
            
            # Test performance
            test_forecast = model_fitted.forecast(train_vars.values[-model_fitted.k_ar:], steps=len(test))
            test_rmse, test_mape, test_mae = calculate_metrics(test['Demand'].values, test_forecast[:, 0])
            
            results['VAR'] = {
                'train': (train_rmse, train_mape, train_mae),
                'val': (val_rmse, val_mape, val_mae),
                'test': (test_rmse, test_mape, test_mae)
            }
            
            # Future forecast
            future_forecast = model_fitted.forecast(train_vars.values[-model_fitted.k_ar:], steps=30)
            future_forecasts['VAR'] = future_forecast[:, 0].tolist()
        except Exception as e:
            results['VAR'] = str(e)

    if 'VARMAX' in selected_models and len(additional_cols) > 0:
        try:
            train_vars = train[['Demand'] + [col + '_mapped' for col in additional_cols]]
            model = VARMAX(train_vars, order=(1, 1)).fit(disp=False)
            
            # Training performance
            train_forecast = model.forecast(steps=len(train))
            train_rmse, train_mape, train_mae = calculate_metrics(train['Demand'].values, train_forecast['Demand'])

            # Validation performance
            val_forecast = model.forecast(steps=len(val))
            validation_predictions['VARMAX'] = {
                'actual': val['Demand'].values,
                'predicted': val_forecast['Demand'],
                'dates': val.index
            }
            
            # Test performance
            test_forecast = model.forecast(steps=len(test))
            test_rmse, test_mape, test_mae = calculate_metrics(test['Demand'].values, test_forecast['Demand'])
            
            results['VARMAX'] = {
                'train': (train_rmse, train_mape, train_mae),
                'val': (val_rmse, val_mape, val_mae),
                'test': (test_rmse, test_mape, test_mae)
            }
            
            # Future forecast
            future_forecast = model.forecast(steps=30)
            future_forecasts['VARMAX'] = future_forecast['Demand'].tolist()
        except Exception as e:
            results['VARMAX'] = str(e)

    # Simple Exponential Smoothing (SES)
    if 'SES' in selected_models:
        try:
            model = SimpleExpSmoothing(train['Demand']).fit()
            
            # Training performance
            train_forecast = model.forecast(steps=len(train))
            train_rmse, train_mape, train_mae = calculate_metrics(train['Demand'].values, train_forecast)
            
            # Validation performance
            val_forecast = model.forecast(steps=len(val))
            validation_predictions['SES'] = {
                'actual': val['Demand'].values,
                'predicted': val_forecast,
                'dates': val.index
            }
            
            # Test performance
            test_forecast = model.forecast(steps=len(test))
            test_rmse, test_mape, test_mae = calculate_metrics(test['Demand'].values, test_forecast)
            
            results['SES'] = {
                'train': (train_rmse, train_mape, train_mae),
                'val': (val_rmse, val_mape, val_mae),
                'test': (test_rmse, test_mape, test_mae)
            }
            
            # Future forecast
            future_forecast = model.forecast(steps=30)
            future_forecasts['SES'] = future_forecast.tolist()
        except Exception as e:
            results['SES'] = str(e)

    # Holt-Winters Exponential Smoothing (HWES)
    if 'HWES' in selected_models:
        try:
            model = ExponentialSmoothing(
                train['Demand'],
                seasonal='add',
                seasonal_periods=seasonal_period
            ).fit()
            
            # Training performance
            train_forecast = model.forecast(steps=len(train))
            train_rmse, train_mape, train_mae = calculate_metrics(train['Demand'].values, train_forecast)
            
            # Validation performance
            val_forecast = model.forecast(steps=len(val))
            validation_predictions['HWES'] = {
                'actual': val['Demand'].values,
                'predicted': val_forecast,
                'dates': val.index
            }
            
            # Test performance
            test_forecast = model.forecast(steps=len(test))
            test_rmse, test_mape, test_mae = calculate_metrics(test['Demand'].values, test_forecast)
            
            results['HWES'] = {
                'train': (train_rmse, train_mape, train_mae),
                'val': (val_rmse, val_mape, val_mae),
                'test': (test_rmse, test_mape, test_mae)
            }
            
            # Future forecast
            future_forecast = model.forecast(steps=30)
            future_forecasts['HWES'] = future_forecast.tolist()
        except Exception as e:
            results['HWES'] = str(e)

    # Prophet Model
    if 'Prophet' in selected_models:
        try:
            prophet_df = train.reset_index().rename(columns={'Date': 'ds', 'Demand': 'y'})
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            model.fit(prophet_df)
            
            # Training performance
            train_forecast = model.predict(prophet_df)['yhat']
            train_rmse, train_mape, train_mae = calculate_metrics(train['Demand'].values, train_forecast)
            
            # Validation performance
            val_df = val.reset_index().rename(columns={'Date': 'ds'})
            val_forecast = model.predict(val_df)['yhat']
            val_rmse, val_mape, val_mae = calculate_metrics(val['Demand'].values, val_forecast)
            
            validation_predictions['Prophet'] = {
                'actual': val['Demand'].values,
                'predicted': val_forecast,
                'dates': val.index
            }
            
            # Test performance
            test_df = test.reset_index().rename(columns={'Date': 'ds'})
            test_forecast = model.predict(test_df)['yhat']
            test_rmse, test_mape, test_mae = calculate_metrics(test['Demand'].values, test_forecast)
            
            results['Prophet'] = {
                'train': (train_rmse, train_mape, train_mae),
                'val': (val_rmse, val_mape, val_mae),
                'test': (test_rmse, test_mape, test_mae)
            }
            
            # Future forecast
            future = model.make_future_dataframe(periods=30)
            future_forecast = model.predict(future)['yhat'][-30:]
            future_forecasts['Prophet'] = future_forecast.tolist()
        except Exception as e:
            results['Prophet'] = str(e)

    # Add visualization of validation results
    st.subheader("Validation Performance")
    
    # Create a plot comparing predictions vs actual values
    fig = go.Figure()
    
    for model_name, pred_data in validation_predictions.items():
        # Plot actual values
        fig.add_trace(go.Scatter(
            x=pred_data['dates'],
            y=pred_data['actual'],
            name='Actual',
            line=dict(color='blue')
        ))
        
        # Plot predicted values
        fig.add_trace(go.Scatter(
            x=pred_data['dates'],
            y=pred_data['predicted'],
            name=f'{model_name} Prediction',
            line=dict(dash='dash')
        ))
    
    fig.update_layout(
        title='Forecast vs Actual Values (Validation Period)',
        xaxis_title='Date',
        yaxis_title='Demand',
        showlegend=True
    )
    
    st.plotly_chart(fig)
    
    # Add validation metrics table
    validation_metrics = pd.DataFrame({
        'Model': [],
        'RMSE': [],
        'MAPE (%)': [],
        'MAE': []
    })
    
    for model_name, pred_data in validation_predictions.items():
        rmse = np.sqrt(mean_squared_error(pred_data['actual'], pred_data['predicted']))
        mape = mean_absolute_percentage_error(pred_data['actual'], pred_data['predicted']) * 100
        mae = mean_absolute_error(pred_data['actual'], pred_data['predicted'])
        
        validation_metrics = pd.concat([validation_metrics, pd.DataFrame({
            'Model': [model_name],
            'RMSE': [round(rmse, 2)],
            'MAPE (%)': [round(mape, 2)],
            'MAE': [round(mae, 2)]
        })])
    
    st.write("### Validation Metrics")
    st.dataframe(validation_metrics)
    
    # Add forecast confidence analysis
    st.write("### Forecast Confidence Analysis")
    for model_name, pred_data in validation_predictions.items():
        mape = mean_absolute_percentage_error(pred_data['actual'], pred_data['predicted']) * 100
        confidence_level = "High" if mape < 10 else "Medium" if mape < 20 else "Low"
        
        st.write(f"**{model_name}**:")
        st.write(f"- Average forecast error (MAPE): {mape:.2f}%")
        st.write(f"- Confidence level: {confidence_level}")
        
        # Calculate percentage of predictions within different error ranges
        errors = np.abs((pred_data['actual'] - pred_data['predicted']) / pred_data['actual'] * 100)
        within_5 = np.mean(errors <= 5) * 100
        within_10 = np.mean(errors <= 10) * 100
        within_20 = np.mean(errors <= 20) * 100
        
        st.write(f"- {within_5:.1f}% of predictions within 5% error")
        st.write(f"- {within_10:.1f}% of predictions within 10% error")
        st.write(f"- {within_20:.1f}% of predictions within 20% error")

    return results, future_forecasts, dates, validation_predictions

def main():
    st.title("Advanced Demand Forecasting Engine")
    st.write("Upload your CSV file and configure the settings to forecast demand.")

    # File upload
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="file_uploader_1")
    if uploaded_file is not None:
        df = load_csv(uploaded_file)
        if df is not None:
            # Display all columns in the dataset
            st.subheader("Columns in the Dataset")
            st.write(df.columns.tolist())

            # Omit columns (if needed)
            st.subheader("Omit Columns")
            omit_cols = st.multiselect("Select columns to omit:", df.columns)
            if omit_cols:
                df = df.drop(omit_cols, axis=1)
                st.write("Updated Columns:", df.columns.tolist())

            # --- MAP COLUMNS SECTION ---
            # First, ask the user to select the date column,
            # so we know which column to exclude from numeric conversion.
            st.subheader("Map Columns")
            date_col = st.selectbox("Select the date column:", df.columns)
            demand_col = st.selectbox("Select the demand column:", df.columns)
            additional_cols = st.multiselect(
                "Select additional time-dependent variables for VAR/VARMAX:",
                df.columns
            )

            # Convert non-date columns to numeric if possible (excluding the date column)
            for col in df.columns:
                if col != date_col and df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except Exception as e:
                        df[col] = df[col].astype(str)

            # Map (rename) columns and normalize the date column using your helper function
            df = map_columns(df, date_col, demand_col, additional_cols)
            # --- END MAP COLUMNS SECTION ---

            if df is not None:
                # Ensure the 'Date' column is a datetime object
                if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

                # Forecasting type selection
                forecast_type = st.radio("Select Forecasting Type:", ("Overall", "Item-wise"))
                if forecast_type == "Item-wise":
                    item_col = st.selectbox("Select the item column:", df.columns)
                    df[item_col] = df[item_col].astype(str)
                    unique_items = df[item_col].unique()

                perform_eda(df)

                # Model selection
                st.subheader("Model Selection")
                selected_models = st.multiselect(
                    "Select models to run:",
                    ["AR", "ARMA", "SARIMA", "VAR", "VARMAX", "SES", "HWES", "Prophet"],
                    default=["SARIMA", "HWES", "Prophet"]
                )

                if st.button("Run Forecast"):
                    with st.spinner("Running models..."):
                        if forecast_type == "Item-wise":
                            results = {}
                            future_forecasts = {}
                            validation_predictions = {}
                            progress_bar = st.progress(0)
                            for i, item in enumerate(unique_items):
                                try:
                                    item_df = df[df[item_col] == item]
                                    if len(item_df) > 0:
                                        item_results, item_future_forecasts, dates, item_validation = forecast_models(
                                            item_df, selected_models, additional_cols
                                        )
                                        results[item] = item_results
                                        future_forecasts[item] = item_future_forecasts
                                        validation_predictions[item] = item_validation
                                    else:
                                        logging.warning(f"No data available for item {item}.")
                                except Exception as e:
                                    st.error(f"Error processing item {item}: {e}")
                                progress_bar.progress((i + 1) / len(unique_items))
                        else:
                            results, future_forecasts, dates, validation_predictions = forecast_models(df, selected_models, additional_cols)
                    
                    # Display forecasts in a table and plot
                    st.subheader("Forecasted Demand")
                    
                    # Get the last date from the dates Series
                    last_date = pd.to_datetime(dates).max()
                    freq = infer_frequency(df, date_col='Date')
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq=freq)
                    future_dates = pd.to_datetime(future_dates)  # Ensure future_dates is a proper DatetimeIndex

                    # Create forecast DataFrame
                    if forecast_type == "Item-wise":
                        all_forecasts = pd.concat([
                            pd.DataFrame({
                                'Item': item,
                                'Date': future_dates.strftime('%Y-%m-%d'),
                                **{model: forecast for model, forecast in future_forecasts[item].items()}
                            }) for item in unique_items
                        ])
                        st.write("### Forecast Table")
                        st.dataframe(all_forecasts)
                        
                        # Plot forecasts for selected item
                        selected_item = st.selectbox("Select item to view forecast plot:", unique_items)
                        item_forecasts = all_forecasts[all_forecasts['Item'] == selected_item]
                        
                        fig = go.Figure()
                        for model in selected_models:
                            if model in item_forecasts.columns:
                                fig.add_trace(go.Scatter(
                                    x=future_dates,
                                    y=item_forecasts[model],
                                    name=f"{model} Forecast",
                                    mode='lines+markers'
                                ))
                        
                        fig.update_layout(
                            title=f'Forecasts for {selected_item}',
                            xaxis_title='Date',
                            yaxis_title='Demand',
                            showlegend=True
                        )
                        st.plotly_chart(fig)
                        
                    else:
                        forecast_df = pd.DataFrame({
                            'Date': future_dates.strftime('%Y-%m-%d'),
                            **{model: forecast for model, forecast in future_forecasts.items()}
                        })
                        
                        # Display forecast table
                        st.write("### Forecast Table")
                        st.dataframe(forecast_df)
                        
                        # Plot forecasts
                        fig = go.Figure()
                        
                        # Add historical data
                        fig.add_trace(go.Scatter(
                            x=dates[-30:],  # Last 30 historical points
                            y=df['Demand'].values[-30:],
                            name='Historical',
                            line=dict(color='black')
                        ))
                        
                        # Add forecasts for each model
                        for model in selected_models:
                            if model in forecast_df.columns:
                                fig.add_trace(go.Scatter(
                                    x=future_dates,
                                    y=forecast_df[model],
                                    name=f"{model} Forecast",
                                    mode='lines+markers'
                                ))
                        
                        fig.update_layout(
                            title='Demand Forecasts by Model',
                            xaxis_title='Date',
                            yaxis_title='Demand',
                            showlegend=True,
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig)

                    # Add download buttons for forecasts
                    st.subheader("Export Forecasts")
                    if forecast_type == "Item-wise":
                        csv = all_forecasts.to_csv(index=False)
                        st.download_button(
                            label="Download All Item Forecasts as CSV",
                            data=csv,
                            file_name='all_item_forecasts.csv',
                            mime='text/csv'
                        )
                    else:
                        csv = forecast_df.to_csv(index=False)
                        st.download_button(
                            label="Download Forecasts as CSV",
                            data=csv,
                            file_name='forecasts.csv',
                            mime='text/csv'
                        )

                    # Export validation data
                    st.subheader("Export Validation Data")
                    if forecast_type == "Item-wise":
                        if validation_predictions:  # Check if we have any validation predictions
                            # Create a list to store validation data for each item
                            validation_dfs = []
                            for item in unique_items:
                                if item in validation_predictions and validation_predictions[item]:
                                    # Get validation data for this item if it exists
                                    item_val_data = pd.DataFrame({
                                        'Item': item,
                                        'Date': validation_predictions[item][next(iter(validation_predictions[item]))]['dates'],
                                        'Actual': validation_predictions[item][next(iter(validation_predictions[item]))]['actual']
                                    })
                                    validation_dfs.append(item_val_data)
                            
                            if validation_dfs:  # If we have any validation data
                                validation_df = pd.concat(validation_dfs, ignore_index=True)
                                csv = validation_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Validation Data as CSV",
                                    data=csv,
                                    file_name='validation_data.csv',
                                    mime='text/csv'
                                )
                            else:
                                st.warning("No validation data available for export.")
                    else:
                        # For overall forecasting (non-item-wise)
                        if validation_predictions:  # Check if we have any validation predictions
                            validation_df = pd.DataFrame({
                                'Date': next(iter(validation_predictions.values()))['dates'],
                                'Actual': next(iter(validation_predictions.values()))['actual']
                            })
                            for model_name, pred_data in validation_predictions.items():
                                validation_df[f'{model_name}_Predicted'] = pred_data['predicted']
                            
                            csv = validation_df.to_csv(index=False)
                            st.download_button(
                                label="Download Validation Data as CSV",
                                data=csv,
                                file_name='validation_data.csv',
                                mime='text/csv'
                            )
                        else:
                            st.warning("No validation data available for export.")

# Run the app
if __name__ == "__main__":
    main()