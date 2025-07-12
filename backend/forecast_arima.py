import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

class ARIMAForecaster:
    """ARIMA-based forecasting model for Indian retail sales"""
    
    def __init__(self):
        self.model = None
        self.fitted_model = None
        self.order = (1, 1, 1)  # Default ARIMA order
        self.seasonal_order = None
        self.is_trained = False
        
    def check_stationarity(self, timeseries):
        """Check if time series is stationary"""
        try:
            result = adfuller(timeseries.dropna())
            return result[1] <= 0.05  # p-value <= 0.05 means stationary
        except:
            return False
    
    def find_best_order(self, timeseries, max_p=3, max_d=2, max_q=3):
        """Find best ARIMA order using AIC"""
        best_aic = float('inf')
        best_order = (1, 1, 1)
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(timeseries, order=(p, d, q))
                        fitted_model = model.fit()
                        aic = fitted_model.aic
                        
                        if aic < best_aic:
                            best_aic = aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        return best_order
    
    def prepare_data(self, historical_data):
        """Prepare time series data for ARIMA"""
        df = pd.DataFrame(historical_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Create time series with proper frequency
        df.set_index('date', inplace=True)
        df.index.freq = 'D'  # Daily frequency
        
        # Handle missing dates
        df = df.asfreq('D', method='pad')
        
        return df['sales']
    
    def detect_seasonality(self, timeseries):
        """Detect seasonal patterns"""
        try:
            if len(timeseries) >= 14:  # Need at least 2 weeks for weekly seasonality
                decomposition = seasonal_decompose(timeseries, model='additive', period=7)
                seasonal_strength = np.var(decomposition.seasonal) / np.var(timeseries)
                return seasonal_strength > 0.1
            return False
        except:
            return False
    
    def train(self, historical_data):
        """Train ARIMA model"""
        try:
            # Prepare data
            timeseries = self.prepare_data(historical_data)
            
            if len(timeseries) < 10:
                raise ValueError("Insufficient data for ARIMA training")
            
            # Check for seasonality
            has_seasonality = self.detect_seasonality(timeseries)
            
            # Find best order
            if len(timeseries) >= 30:  # Enough data for parameter optimization
                self.order = self.find_best_order(timeseries)
            
            # Fit ARIMA model
            if has_seasonality and len(timeseries) >= 21:
                # Use SARIMA for seasonal data
                self.seasonal_order = (1, 1, 1, 7)  # Weekly seasonality
                self.model = ARIMA(timeseries, order=self.order, seasonal_order=self.seasonal_order)
            else:
                # Use simple ARIMA
                self.model = ARIMA(timeseries, order=self.order)
            
            self.fitted_model = self.model.fit()
            self.is_trained = True
            
            return True
            
        except Exception as e:
            print(f"ARIMA training error: {e}")
            return False
    
    def forecast(self, historical_data, forecast_days=7):
        """Generate forecast using ARIMA"""
        try:
            # Train model if not already trained
            if not self.is_trained:
                if not self.train(historical_data):
                    return self._fallback_forecast(historical_data, forecast_days)
            
            # Generate forecast
            forecast_result = self.fitted_model.forecast(steps=forecast_days)
            forecast_values = forecast_result.tolist()
            
            # Get confidence intervals
            confidence_intervals = self.fitted_model.get_forecast(steps=forecast_days).conf_int()
            
            # Generate forecast dates
            last_date = datetime.strptime(historical_data[-1]['date'], '%Y-%m-%d')
            forecast_dates = []
            
            for i in range(forecast_days):
                future_date = last_date + timedelta(days=i+1)
                forecast_dates.append(future_date.strftime('%Y-%m-%d'))
            
            # Ensure non-negative values
            forecast_values = [max(0, int(value)) for value in forecast_values]
            
            # Format confidence intervals
            conf_intervals = []
            for i in range(len(confidence_intervals)):
                lower = max(0, int(confidence_intervals.iloc[i, 0]))
                upper = int(confidence_intervals.iloc[i, 1])
                conf_intervals.append({
                    'lower': lower,
                    'upper': upper
                })
            
            return {
                'model': 'ARIMA',
                'dates': forecast_dates,
                'values': forecast_values,
                'confidence_intervals': conf_intervals,
                'parameters': {
                    'order': self.order,
                    'seasonal_order': self.seasonal_order,
                    'aic': round(self.fitted_model.aic, 2) if self.fitted_model else None
                }
            }
            
        except Exception as e:
            print(f"ARIMA forecast error: {e}")
            return self._fallback_forecast(historical_data, forecast_days)
    
    def _fallback_forecast(self, historical_data, forecast_days):
        """Simple fallback forecast when ARIMA fails"""
        # Use moving average with trend
        sales_values = [item['sales'] for item in historical_data]
        
        # Calculate moving average and trend
        if len(sales_values) >= 7:
            ma_7 = np.mean(sales_values[-7:])
            trend = (sales_values[-1] - sales_values[-7]) / 7
        else:
            ma_7 = np.mean(sales_values)
            trend = 0
        
        # Generate forecast
        forecast_dates = []
        forecast_values = []
        
        last_date = datetime.strptime(historical_data[-1]['date'], '%Y-%m-%d')
        
        for i in range(forecast_days):
            future_date = last_date + timedelta(days=i+1)
            predicted_value = max(0, int(ma_7 + trend * (i+1)))
            
            forecast_dates.append(future_date.strftime('%Y-%m-%d'))
            forecast_values.append(predicted_value)
        
        # Simple confidence intervals
        std_dev = np.std(sales_values)
        confidence_intervals = []
        for value in forecast_values:
            lower = max(0, int(value - 1.96 * std_dev))
            upper = int(value + 1.96 * std_dev)
            confidence_intervals.append({
                'lower': lower,
                'upper': upper
            })
        
        return {
            'model': 'ARIMA (Fallback)',
            'dates': forecast_dates,
            'values': forecast_values,
            'confidence_intervals': confidence_intervals,
            'parameters': {'fallback': True}
        }
    
    def get_model_summary(self):
        """Get model summary statistics"""
        if self.fitted_model:
            return {
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'order': self.order,
                'seasonal_order': self.seasonal_order
            }
        return None
    
    def save_model(self, filepath):
        """Save trained model"""
        if self.is_trained:
            model_data = {
                'fitted_model': self.fitted_model,
                'order': self.order,
                'seasonal_order': self.seasonal_order
            }
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            return True
        return False
    
    def load_model(self, filepath):
        """Load trained model"""
        try:
            import pickle
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.fitted_model = model_data['fitted_model']
            self.order = model_data['order']
            self.seasonal_order = model_data['seasonal_order']
            self.is_trained = True
            return True
        except:
            return False