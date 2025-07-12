import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import warnings
warnings.filterwarnings('ignore')

class XGBoostForecaster:
    """XGBoost-based forecasting model for Indian retail sales"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        
    def create_features(self, data):
        """Create features for XGBoost model"""
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Time-based features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        
        # Indian festival features
        df['is_diwali_season'] = ((df['month'] == 10) | (df['month'] == 11)).astype(int)
        df['is_dussehra_season'] = (df['month'] == 9).astype(int)
        df['is_holi_season'] = ((df['month'] == 3) | (df['month'] == 4)).astype(int)
        df['is_monsoon'] = ((df['month'] >= 6) & (df['month'] <= 8)).astype(int)
        
        # Lag features
        df['sales_lag_1'] = df['sales'].shift(1)
        df['sales_lag_7'] = df['sales'].shift(7)
        df['sales_lag_30'] = df['sales'].shift(30)
        
        # Rolling statistics
        df['sales_rolling_mean_7'] = df['sales'].rolling(window=7).mean()
        df['sales_rolling_std_7'] = df['sales'].rolling(window=7).std()
        df['sales_rolling_mean_30'] = df['sales'].rolling(window=30).mean()
        
        # Trend features
        df['sales_diff_1'] = df['sales'].diff(1)
        df['sales_diff_7'] = df['sales'].diff(7)
        
        # Store and category encoding
        store_mapping = {
            'mumbai': 1, 'delhi': 2, 'bangalore': 3, 
            'chennai': 4, 'hyderabad': 5, 'pune': 6, 'all': 0
        }
        category_mapping = {
            'clothing': 1, 'electronics': 2, 'groceries': 3,
            'home': 4, 'beauty': 5, 'all': 0
        }
        
        df['store_encoded'] = df['store'].map(store_mapping).fillna(0)
        df['category_encoded'] = df['category'].map(category_mapping).fillna(0)
        
        # Fill missing values
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    def prepare_training_data(self, df):
        """Prepare features and target for training"""
        feature_cols = [
            'year', 'month', 'day', 'dayofweek', 'quarter', 'is_weekend',
            'is_diwali_season', 'is_dussehra_season', 'is_holi_season', 'is_monsoon',
            'sales_lag_1', 'sales_lag_7', 'sales_lag_30',
            'sales_rolling_mean_7', 'sales_rolling_std_7', 'sales_rolling_mean_30',
            'sales_diff_1', 'sales_diff_7', 'store_encoded', 'category_encoded'
        ]
        
        # Remove rows with NaN values (due to lag features)
        df_clean = df.dropna()
        
        X = df_clean[feature_cols]
        y = df_clean['sales']
        
        self.feature_columns = feature_cols
        return X, y
    
    def train(self, historical_data):
        """Train the XGBoost model"""
        try:
            # Create features
            df = self.create_features(historical_data)
            
            # Prepare training data
            X, y = self.prepare_training_data(df)
            
            if len(X) < 10:  # Not enough data for training
                raise ValueError("Insufficient data for training")
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train XGBoost model
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            return True
            
        except Exception as e:
            print(f"Training error: {e}")
            return False
    
    def forecast(self, historical_data, forecast_days=7):
        """Generate forecast using XGBoost"""
        try:
            # Train model if not already trained
            if not self.is_trained:
                if not self.train(historical_data):
                    return self._fallback_forecast(historical_data, forecast_days)
            
            # Create features for historical data
            df = self.create_features(historical_data)
            
            # Get the last date and sales values
            last_date = pd.to_datetime(df['date'].iloc[-1])
            last_sales = df['sales'].iloc[-1]
            
            # Generate forecast
            forecast_dates = []
            forecast_values = []
            
            for i in range(forecast_days):
                # Generate future date
                future_date = last_date + timedelta(days=i+1)
                
                # Create features for future date
                future_features = self._create_future_features(
                    df, future_date, last_sales, forecast_values
                )
                
                # Scale features
                future_features_scaled = self.scaler.transform([future_features])
                
                # Predict
                predicted_sales = self.model.predict(future_features_scaled)[0]
                predicted_sales = max(0, predicted_sales)  # Ensure non-negative
                
                forecast_dates.append(future_date.strftime('%Y-%m-%d'))
                forecast_values.append(int(predicted_sales))
                
                # Update last_sales for next iteration
                last_sales = predicted_sales
            
            # Calculate confidence intervals (simple approach)
            confidence_intervals = self._calculate_confidence_intervals(
                forecast_values, historical_data
            )
            
            return {
                'model': 'XGBoost',
                'dates': forecast_dates,
                'values': forecast_values,
                'confidence_intervals': confidence_intervals,
                'parameters': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1
                }
            }
            
        except Exception as e:
            print(f"Forecast error: {e}")
            return self._fallback_forecast(historical_data, forecast_days)
    
    def _create_future_features(self, df, future_date, last_sales, forecast_values):
        """Create features for future date"""
        # Time-based features
        features = [
            future_date.year,
            future_date.month,
            future_date.day,
            future_date.weekday(),
            future_date.quarter,
            int(future_date.weekday() >= 5),  # is_weekend
            int(future_date.month in [10, 11]),  # is_diwali_season
            int(future_date.month == 9),  # is_dussehra_season
            int(future_date.month in [3, 4]),  # is_holi_season
            int(6 <= future_date.month <= 8),  # is_monsoon
        ]
        
        # Lag features (use last known values or recent forecasts)
        if len(forecast_values) > 0:
            features.append(forecast_values[-1])  # sales_lag_1
        else:
            features.append(last_sales)  # sales_lag_1
        
        # For 7-day and 30-day lags, use historical data
        if len(df) >= 7:
            features.append(df['sales'].iloc[-7])  # sales_lag_7
        else:
            features.append(df['sales'].iloc[-1])
        
        if len(df) >= 30:
            features.append(df['sales'].iloc[-30])  # sales_lag_30
        else:
            features.append(df['sales'].mean())
        
        # Rolling statistics (use recent historical data)
        recent_sales = df['sales'].iloc[-7:].tolist()
        if len(forecast_values) > 0:
            recent_sales.extend(forecast_values)
        
        features.extend([
            np.mean(recent_sales[-7:]),  # sales_rolling_mean_7
            np.std(recent_sales[-7:]),   # sales_rolling_std_7
            np.mean(recent_sales[-30:]) if len(recent_sales) >= 30 else np.mean(recent_sales),  # sales_rolling_mean_30
        ])
        
        # Difference features
        if len(forecast_values) > 0:
            features.append(forecast_values[-1] - last_sales)  # sales_diff_1
        else:
            features.append(0)  # sales_diff_1
        
        if len(df) >= 7:
            features.append(last_sales - df['sales'].iloc[-7])  # sales_diff_7
        else:
            features.append(0)
        
        # Store and category encoding (use from historical data)
        features.extend([
            df['store_encoded'].iloc[-1],
            df['category_encoded'].iloc[-1]
        ])
        
        return features
    
    def _calculate_confidence_intervals(self, forecast_values, historical_data):
        """Calculate simple confidence intervals"""
        historical_sales = [item['sales'] for item in historical_data]
        std_dev = np.std(historical_sales)
        
        confidence_intervals = []
        for value in forecast_values:
            lower = max(0, value - 1.96 * std_dev)
            upper = value + 1.96 * std_dev
            confidence_intervals.append({
                'lower': int(lower),
                'upper': int(upper)
            })
        
        return confidence_intervals
    
    def _fallback_forecast(self, historical_data, forecast_days):
        """Simple fallback forecast when XGBoost fails"""
        # Use simple trend-based forecasting
        sales_values = [item['sales'] for item in historical_data]
        
        # Calculate trend
        if len(sales_values) >= 7:
            recent_avg = np.mean(sales_values[-7:])
            trend = (sales_values[-1] - sales_values[-7]) / 7
        else:
            recent_avg = np.mean(sales_values)
            trend = 0
        
        # Generate forecast
        forecast_dates = []
        forecast_values = []
        
        last_date = datetime.strptime(historical_data[-1]['date'], '%Y-%m-%d')
        
        for i in range(forecast_days):
            future_date = last_date + timedelta(days=i+1)
            predicted_value = max(0, int(recent_avg + trend * (i+1)))
            
            forecast_dates.append(future_date.strftime('%Y-%m-%d'))
            forecast_values.append(predicted_value)
        
        return {
            'model': 'XGBoost (Fallback)',
            'dates': forecast_dates,
            'values': forecast_values,
            'confidence_intervals': [{'lower': int(v*0.9), 'upper': int(v*1.1)} for v in forecast_values],
            'parameters': {'fallback': True}
        }
    
    def save_model(self, filepath):
        """Save trained model"""
        if self.is_trained:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            return True
        return False
    
    def load_model(self, filepath):
        """Load trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = True
            return True
        except:
            return False