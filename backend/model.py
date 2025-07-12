import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
except ImportError:
    from prophet import Prophet

import pickle
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

class RetailSalesForecaster:
    """
    A comprehensive retail sales forecasting system for Indian market
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        
    def load_data(self, train_path='data/train.csv', store_path='data/store.csv'):
        """Load and prepare Indian retail sales data"""
        try:
            # Load data
            self.train_df = pd.read_csv(train_path)
            self.store_df = pd.read_csv(store_path)
            
            # Merge store information
            self.df = pd.merge(self.train_df, self.store_df, on='Store', how='left')
            
            print(f"Data loaded successfully! Shape: {self.df.shape}")
            return True
            
        except FileNotFoundError:
            print("Data files not found. Generating sample Indian retail data...")
            self.generate_sample_data()
            return True
    
    def generate_sample_data(self):
        """Generate sample Indian retail sales data"""
        # Generate 2 years of daily data for 10 stores
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2023, 12, 31)
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        stores = range(1, 11)
        data = []
        
        for store in stores:
            for date in date_range:
                # Base sales patterns for Indian market
                base_sales = np.random.normal(45000, 12000)
                
                # Day of week effect
                day_of_week = date.weekday()
                if day_of_week in [5, 6]:  # Weekend
                    base_sales *= 1.3
                
                # Seasonal patterns
                month = date.month
                if month in [10, 11, 12]:  # Festival season
                    base_sales *= 1.6
                elif month in [3, 4, 5]:  # Summer
                    base_sales *= 0.9
                elif month in [6, 7, 8, 9]:  # Monsoon
                    base_sales *= 0.85
                
                # Store type effect
                store_type = ['Supermarket', 'Hypermarket', 'Mall', 'Standalone'][store % 4]
                if store_type == 'Hypermarket':
                    base_sales *= 1.4
                elif store_type == 'Mall':
                    base_sales *= 1.2
                elif store_type == 'Standalone':
                    base_sales *= 0.8
                
                # Ensure positive sales
                sales = max(base_sales, 5000)
                
                # Create features
                data.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    'Store': store,
                    'Sales': round(sales, 2),
                    'Customers': int(sales / 450),  # Approx customers
                    'Open': 1 if date.weekday() != 6 else 0,  # Closed on Sunday
                    'Promo': 1 if np.random.random() < 0.3 else 0,  # 30% promo days
                    'StateHoliday': 1 if date.month == 8 and date.day == 15 else 0,  # Independence Day
                    'SchoolHoliday': 1 if month in [5, 6, 12] else 0,  # School holidays
                    'StoreType': store_type,
                    'Assortment': ['Basic', 'Extended', 'Extra'][store % 3],
                    'CompetitionDistance': np.random.randint(100, 5000),
                    'Year': date.year,
                    'Month': date.month,
                    'DayOfWeek': date.weekday() + 1,
                    'WeekOfYear': date.isocalendar()[1]
                })
        
        self.df = pd.DataFrame(data)
        
        # Save sample data
        os.makedirs('data', exist_ok=True)
        self.df.to_csv('data/cleaned_data.csv', index=False)
        
        # Create store data
        store_data = []
        for store in stores:
            store_data.append({
                'Store': store,
                'StoreType': ['Supermarket', 'Hypermarket', 'Mall', 'Standalone'][store % 4],
                'Assortment': ['Basic', 'Extended', 'Extra'][store % 3],
                'CompetitionDistance': np.random.randint(100, 5000)
            })
        
        store_df = pd.DataFrame(store_data)
        store_df.to_csv('data/store.csv', index=False)
        
        print(f"Sample data generated! Shape: {self.df.shape}")
    
    def feature_engineering(self):
        """Create features for Indian retail patterns"""
        # Convert date
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Time-based features
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        self.df['DayOfWeek'] = self.df['Date'].dt.dayofweek
        self.df['WeekOfYear'] = self.df['Date'].dt.isocalendar().week
        self.df['DayOfMonth'] = self.df['Date'].dt.day
        self.df['Quarter'] = self.df['Date'].dt.quarter
        
        # Indian festival seasons
        self.df['IsFestivalSeason'] = self.df['Month'].isin([10, 11, 12]).astype(int)
        self.df['IsSummerSeason'] = self.df['Month'].isin([3, 4, 5]).astype(int)
        self.df['IsMonsoonSeason'] = self.df['Month'].isin([6, 7, 8, 9]).astype(int)
        
        # Weekend/weekday
        self.df['IsWeekend'] = self.df['DayOfWeek'].isin([5, 6]).astype(int)
        
        # Lag features
        self.df = self.df.sort_values(['Store', 'Date'])
        for lag in [1, 7, 30]:
            self.df[f'Sales_Lag_{lag}'] = self.df.groupby('Store')['Sales'].shift(lag)
        
        # Rolling features
        for window in [7, 30]:
            self.df[f'Sales_Rolling_Mean_{window}'] = self.df.groupby('Store')['Sales'].rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
            self.df[f'Sales_Rolling_Std_{window}'] = self.df.groupby('Store')['Sales'].rolling(window=window, min_periods=1).std().reset_index(0, drop=True)
        
        # Encode categorical variables
        from sklearn.preprocessing import LabelEncoder
        
        categorical_features = ['StoreType', 'Assortment']
        for feature in categorical_features:
            if feature in self.df.columns:
                le = LabelEncoder()
                self.df[f'{feature}_Encoded'] = le.fit_transform(self.df[feature].astype(str))
        
        # Drop rows with NaN values created by lag features
        self.df = self.df.dropna()
        
        print(f"Feature engineering completed! Final shape: {self.df.shape}")
    
    def train_xgboost_model(self):
        """Train XGBoost model for sales forecasting"""
        # Select features
        feature_columns = [
            'Store', 'DayOfWeek', 'Month', 'Year', 'WeekOfYear', 'DayOfMonth', 'Quarter',
            'IsFestivalSeason', 'IsSummerSeason', 'IsMonsoonSeason', 'IsWeekend',
            'Promo', 'StateHoliday', 'SchoolHoliday', 'CompetitionDistance',
            'Sales_Lag_1', 'Sales_Lag_7', 'Sales_Lag_30',
            'Sales_Rolling_Mean_7', 'Sales_Rolling_Mean_30',
            'Sales_Rolling_Std_7', 'Sales_Rolling_Std_30'
        ]
        
        # Add encoded categorical features
        if 'StoreType_Encoded' in self.df.columns:
            feature_columns.append('StoreType_Encoded')
        if 'Assortment_Encoded' in self.df.columns:
            feature_columns.append('Assortment_Encoded')
        
        # Filter available columns
        available_features = [col for col in feature_columns if col in self.df.columns]
        
        X = self.df[available_features]
        y = self.df['Sales']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train XGBoost model
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        xgb_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = xgb_model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"XGBoost Model Performance:")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        
        # Store model and metrics
        self.models['xgboost'] = xgb_model
        self.feature_importance['xgboost'] = dict(zip(available_features, xgb_model.feature_importances_))
        self.performance_metrics['xgboost'] = {'rmse': rmse, 'mae': mae}
        
        # Save model
        os.makedirs('models', exist_ok=True)
        with open('models/xgboost_model.pkl', 'wb') as f:
            pickle.dump(xgb_model, f)
        
        print("XGBoost model trained and saved!")
        
        
        
        
    def train_prophet_model(self):
        """Train Facebook Prophet model"""
        # Prepare data for Prophet
        store_1_data = self.df[self.df['Store'] == 1][['Date', 'Sales']].copy()
        store_1_data.columns = ['ds', 'y']
        
        # Create Prophet model
        prophet_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        
        # Add Indian holidays
        prophet_model.add_country_holidays(country_name='IN')
        
        # Add custom seasonalities
        prophet_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        
        # Fit model
        prophet_model.fit(store_1_data)
        
        # Make future dataframe
        future = prophet_model.make_future_dataframe(periods=30)
        forecast = prophet_model.predict(future)
        
        # Calculate simple metrics (using last 30 days)
        test_data = store_1_data.tail(30)
        test_forecast = forecast.tail(30)
        
        mae = mean_absolute_error(test_data['y'], test_forecast['yhat'])
        rmse = np.sqrt(mean_squared_error(test_data['y'], test_forecast['yhat']))
        
        print(f"Prophet Model Performance:")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        
        # Store model and metrics
        self.models['prophet'] = prophet_model
        self.performance_metrics['prophet'] = {'rmse': rmse, 'mae': mae}
        
        # Save model
        with open('models/prophet_model.pkl', 'wb') as f:
            pickle.dump(prophet_model, f)
        
        print("Prophet model trained and saved!")
        
        
        
    
    def train_arima_model(self):
        """Train ARIMA model"""
        # Prepare data for ARIMA
        store_1_data = self.df[self.df['Store'] == 1]['Sales'].values
        
        
        # Fit ARIMA model
        arima_model = ARIMA(store_1_data, order=(1, 1, 1))
        arima_fitted = arima_model.fit()
        
        # Make predictions
        forecast = arima_fitted.forecast(steps=30)
        
        # Calculate metrics (using in-sample fit)
        fitted_values = arima_fitted.fittedvalues
        residuals = store_1_data[1:] - fitted_values  # Skip first value due to differencing
        
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals**2))
        
        print(f"ARIMA Model Performance:")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        
        # Store model and metrics
        self.models['arima'] = arima_fitted
        self.performance_metrics['arima'] = {'rmse': rmse, 'mae': mae}
        
        # Save model
        with open('models/arima_model.pkl', 'wb') as f:
            pickle.dump(arima_fitted, f)
        
        print("ARIMA model trained and saved!")
        
        
    
    def train_all_models(self):
        """Train all forecasting models"""
        print("Starting model training...")
        
        # Load and prepare data
        self.load_data()
        self.feature_engineering()
        
        # Train models
        self.train_xgboost_model()
        self.train_prophet_model()
        self.train_arima_model()
        
        print("\nAll models trained successfully!")
        print("\nModel Performance Summary:")
        for model_name, metrics in self.performance_metrics.items():
            print(f"{model_name.upper()}: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}")
    
    def plot_feature_importance(self):
        """Plot feature importance for XGBoost model"""
        if 'xgboost' in self.feature_importance:
            plt.figure(figsize=(10, 8))
            
            # Sort features by importance
            sorted_features = sorted(self.feature_importance['xgboost'].items(), 
                                   key=lambda x: x[1], reverse=True)
            
            features = [item[0] for item in sorted_features[:15]]  # Top 15 features
            importances = [item[1] for item in sorted_features[:15]]
            
            plt.barh(features, importances)
            plt.xlabel('Feature Importance')
            plt.title('XGBoost Feature Importance - Indian Retail Sales')
            plt.tight_layout()
            plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("Feature importance plot saved!")

if __name__ == "__main__":
    # Initialize forecaster
    forecaster = RetailSalesForecaster()
    
    # Train all models
    forecaster.train_all_models()
    
    # Plot feature importance
    forecaster.plot_feature_importance()
    
    print("\nModel training completed! Models saved in 'models/' directory.")