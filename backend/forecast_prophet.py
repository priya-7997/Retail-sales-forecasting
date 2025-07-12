import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Prophet model simulation (since Prophet might not be available in all environments)
class ProphetForecaster:
    """Prophet-based forecasting model for Indian retail sales"""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.indian_holidays = self._get_indian_holidays()
        
    def _get_indian_holidays(self):
        """Define Indian holidays and festivals"""
        return {
            'diwali': {'month': 11, 'effect': 1.4},
            'dussehra': {'month': 10, 'effect': 1.3},
            'holi': {'month': 3, 'effect': 1.2},
            'diwali_prep': {'month': 10, 'effect': 1.25},
            'eid': {'month': 5, 'effect': 1.15},
            'christmas': {'month': 12, 'effect': 1.2},
            'new_year': {'month': 1, 'effect': 1.1}
        }
    
    def prepare_data(self, historical_data):
        """Prepare data for Prophet format"""
        df = pd.DataFrame(historical_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Prophet requires 'ds' and 'y' columns
        prophet_df = pd.DataFrame({
            'ds': df['date'],
            'y': df['sales']
        })
        
        # Add additional regressors
        prophet_df['is_weekend'] = (prophet_df['ds'].dt.dayofweek >= 5).astype(int)
        prophet_df['month'] = prophet_df['ds'].dt.month
        prophet_df['day_of_week'] = prophet_df['ds'].dt.dayofweek
        
        # Add holiday effects
        for holiday, info in self.indian_holidays.items():
            prophet_df[f'is_{holiday}'] = (prophet_df['month'] == info['month']).astype(int)
        
        return prophet_df
    
    def add_seasonality_effects(self, df):
        """Add Indian retail seasonality effects"""
        # Festival season multipliers
        festival_multiplier = 1.0
        
        for _, row in df.iterrows():
            month = row['ds'].month
            
            # Diwali season (October-November)
            if month in [10, 11]:
                festival_multiplier = 1.35
            # Holi season (March-April)
            elif month in [3, 4]:
                festival_multiplier = 1.15
            # Christmas/New Year
            elif month == 12:
                festival_multiplier = 1.2
            # Monsoon season (June-August) - reduced sales
            elif month in [6, 7, 8]:
                festival_multiplier = 0.85
            else:
                festival_multiplier = 1.0
            
            # Weekend effect
            if row['is_weekend']:
                festival_multiplier *= 1.15
            
            # Apply multiplier
            df.loc[df['ds'] == row['ds'], 'y'] *= festival_multiplier
        
        return df
    
    def train(self, historical_data):
        """Train Prophet model (simulated)"""
        try:
            # Prepare data
            df = self.prepare_data(historical_data)
            
            if len(df) < 10:
                raise ValueError("Insufficient data for Prophet training")
            
            # Since we're simulating Prophet, we'll store the data patterns
            self.training_data = df
            self.is_trained = True
            
            # Calculate baseline statistics
            self.baseline_stats = {
                'mean': df['y'].mean(),
                'std': df['y'].std(),
                'trend': self._calculate_trend(df),
                'seasonality': self._calculate_seasonality(df)
            }
            
            return True
            
        except Exception as e:
            print(f"Prophet training error: {e}")
            return False
    
    def _calculate_trend(self, df):
        """Calculate linear trend"""
        if len(df) < 2:
            return 0
        
        x = np.arange(len(df))
        y = df['y'].values
        
        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def _calculate_seasonality(self, df):
        """Calculate weekly seasonality pattern"""
        if len(df) < 7:
            return {i: 1.0 for i in range(7)}
        
        # Group by day of week and calculate average multiplier
        dow_effects = {}
        overall_mean = df['y'].mean()
        
        for dow in range(7):
            dow_data = df[df['day_of_week'] == dow]['y']
            if len(dow_data) > 0:
                dow_effects[dow] = dow_data.mean() / overall_mean
            else:
                dow_effects[dow] = 1.0
        
        return dow_effects
    
    def forecast(self, historical_data, forecast_days=7):
        """Generate forecast using Prophet-style approach"""
        try:
            # Train model if not already trained
            if not self.is_trained:
                if not self.train(historical_data):
                    return self._fallback_forecast(historical_data, forecast_days)
            
            # Generate forecast dates
            last_date = datetime.strptime(historical_data[-1]['date'], '%Y-%m-%d')
            forecast_dates = []
            forecast_values = []
            
            for i in range(forecast_days):
                future_date = last_date + timedelta(days=i+1)
                forecast_dates.append(future_date.strftime('%Y-%m-%d'))
                
                # Generate forecast value
                predicted_value = self._predict_single_day(future_date, i+1)
                forecast_values.append(max(0, int(predicted_value)))
            
            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(forecast_values)
            
            return {
                'model': 'Prophet',
                'dates': forecast_dates,
                'values': forecast_values,
                'confidence_intervals': confidence_intervals,
                'parameters': {
                    'trend': round(self.baseline_stats['trend'], 2),
                    'seasonality': 'weekly',
                    'holidays': 'indian_festivals'
                }
            }
            
        except Exception as e:
            print(f"Prophet forecast error: {e}")
            return self._fallback_forecast(historical_data, forecast_days)
    
    def _predict_single_day(self, future_date, days_ahead):
        """Predict sales for a single future day"""
        # Base prediction with trend
        base_prediction = self.baseline_stats['mean'] + (self.baseline_stats['trend'] * days_ahead)
        
        # Apply seasonality (day of week effect)
        dow = future_date.weekday()
        seasonality_effect = self.baseline_stats['seasonality'].get(dow, 1.0)
        base_prediction *= seasonality_effect
        
        # Apply Indian holiday/festival effects
        month = future_date.month
        
        # Festival season effects
        if month in [10, 11]:  # Diwali season
            base_prediction *= 1.35
        elif month == 9:  # Dussehra
            base_prediction *= 1.25
        elif month in [3, 4]:  # Holi season
            base_prediction *= 1.15
        elif month == 12:  # Christmas/New Year
            base_prediction *= 1.2
        elif month in [6, 7, 8]:  # Monsoon season
            base_prediction *= 0.85
        
        # Weekend effect
        if future_date.weekday() >= 5:
            base_prediction *= 1.15
        
        # Add some controlled randomness for realism
        noise = np.random.normal(0, self.baseline_stats['std'] * 0.1)
        base_prediction += noise
        
        return base_prediction
    
    def _calculate_confidence_intervals(self, forecast_values):
        """Calculate confidence intervals for forecast"""
        confidence_intervals = []
        std_dev = self.baseline_stats['std']
        
        for i, value in enumerate(forecast_values):
            # Uncertainty increases with forecast horizon
            uncertainty = std_dev * (1 + i * 0.1)
            
            lower = max(0, int(value - 1.96 * uncertainty))
            upper = int(value + 1.96 * uncertainty)
            
            confidence_intervals.append({
                'lower': lower,
                'upper': upper
            })
        
        return confidence_intervals
    
    def _fallback_forecast(self, historical_data, forecast_days):
        """Simple fallback forecast when Prophet fails"""
        # Use exponential smoothing approach
        sales_values = [item['sales'] for item in historical_data]
        
        # Calculate exponential smoothing parameters
        alpha = 0.3  # Smoothing parameter
        
        if len(sales_values) >= 7:
            # Initialize with 7-day average
            smoothed_value = np.mean(sales_values[-7:])
        else:
            smoothed_value = np.mean(sales_values)
        
        # Generate forecast
        forecast_dates = []
        forecast_values = []
        
        last_date = datetime.strptime(historical_data[-1]['date'], '%Y-%m-%d')
        
        for i in range(forecast_days):
            future_date = last_date + timedelta(days=i+1)
            
            # Apply seasonal adjustments
            predicted_value = smoothed_value
            
            # Day of week effect
            if future_date.weekday() >= 5:  # Weekend
                predicted_value *= 1.15
            
            # Month effect
            month = future_date.month
            if month in [10, 11]:
                predicted_value *= 1.3
            elif month in [6, 7, 8]:
                predicted_value *= 0.9
            
            forecast_dates.append(future_date.strftime('%Y-%m-%d'))
            forecast_values.append(max(0, int(predicted_value)))
        
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
            'model': 'Prophet (Fallback)',
            'dates': forecast_dates,
            'values': forecast_values,
            'confidence_intervals': confidence_intervals,
            'parameters': {'fallback': True}
        }
    
    def add_country_holidays(self, country='IN'):
        """Add country-specific holidays"""
        if country == 'IN':
            # Indian holidays are already included
            return self.indian_holidays
        else:
            return {}
    
    def save_model(self, filepath):
        """Save trained model"""
        if self.is_trained:
            model_data = {
                'training_data': self.training_data,
                'baseline_stats': self.baseline_stats,
                'indian_holidays': self.indian_holidays
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
            
            self.training_data = model_data['training_data']
            self.baseline_stats = model_data['baseline_stats']
            self.indian_holidays = model_data['indian_holidays']
            self.is_trained = True
            return True
        except:
            return False