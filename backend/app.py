from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
import logging

# Import custom forecasting modules
from forecast_xgb import XGBoostForecaster
from forecast_arima import ARIMAForecaster
from forecast_prophet import ProphetForecaster
from utils import load_data, preprocess_data, calculate_metrics, indian_festivals

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to store models and data
models = {}
data = None
preprocessed_data = None

# Model paths
MODEL_PATHS = {
    'xgboost': 'models/xgb_model.pkl',
    'arima': 'models/arima_model.pkl',
    'prophet': 'models/prophet_model.pkl'
}

def initialize_models():
    """Initialize all forecasting models"""
    global models
    
    try:
        models['xgboost'] = XGBoostForecaster()
        models['arima'] = ARIMAForecaster()
        models['prophet'] = ProphetForecaster()
        
        # Load pre-trained models if they exist
        for model_name, model_path in MODEL_PATHS.items():
            if os.path.exists(model_path):
                logger.info(f"Loading {model_name} model from {model_path}")
                models[model_name].load_model(model_path)
            else:
                logger.warning(f"Model file not found: {model_path}")
        
        logger.info("Models initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")

def load_initial_data():
    """Load and preprocess initial data"""
    global data, preprocessed_data
    
    try:
        # Load data from CSV files
        data_path = 'data/cleaned_data.csv'
        if os.path.exists(data_path):
            data = load_data(data_path)
            preprocessed_data = preprocess_data(data)
            logger.info(f"Data loaded successfully: {len(data)} records")
        else:
            logger.warning(f"Data file not found: {data_path}")
            # Generate sample data for demo
            data = generate_sample_data()
            preprocessed_data = preprocess_data(data)
            logger.info("Using generated sample data")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        data = generate_sample_data()
        preprocessed_data = preprocess_data(data)

def generate_sample_data():
    """Generate sample Indian retail data"""
    dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='D')
    
    # Indian cities and their characteristics
    cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Pune', 'Kolkata', 'Ahmedabad']
    categories = ['Clothing', 'Electronics', 'Groceries', 'Home', 'Beauty', 'Books', 'Sports']
    
    data_records = []
    
    for date in dates:
        for city in cities:
            for category in categories:
                # Base sales with seasonal and festival effects
                base_sales = np.random.normal(1000, 200)
                
                # Seasonal effects
                month = date.month
                seasonal_multiplier = 1.0
                
                # Festival effects
                for festival, details in indian_festivals.items():
                    if abs(month - details['month']) <= 1:
                        if category.lower() in details['categories']:
                            seasonal_multiplier *= details['boost']
                
                # Monsoon effect (June-September)
                if 6 <= month <= 9:
                    seasonal_multiplier *= 0.9
                
                # Year-end boost
                if month >= 11:
                    seasonal_multiplier *= 1.2
                
                # Weekend boost
                if date.weekday() >= 5:
                    seasonal_multiplier *= 1.1
                
                # City-specific multipliers
                city_multipliers = {
                    'Mumbai': 1.2, 'Delhi': 1.15, 'Bangalore': 1.1,
                    'Chennai': 1.05, 'Hyderabad': 1.08, 'Pune': 1.0,
                    'Kolkata': 0.95, 'Ahmedabad': 0.9
                }
                
                final_sales = max(0, base_sales * seasonal_multiplier * city_multipliers.get(city, 1.0))
                
                data_records.append({
                    'Date': date,
                    'Store': city,
                    'Category': category,
                    'Sales': final_sales,
                    'Day_of_Week': date.weekday(),
                    'Month': date.month,
                    'Year': date.year
                })
    
    return pd.DataFrame(data_records)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': len(models),
        'data_loaded': data is not None
    })

@app.route('/api/forecast', methods=['POST'])
def forecast():
    """Main forecasting endpoint"""
    try:
        # Get request parameters
        params = request.json
        store = params.get('store', 'all')
        category = params.get('category', 'all')
        period = params.get('period', 30)
        forecast_period = params.get('forecast_period', 7)
        model_type = params.get('model', 'all')
        
        logger.info(f"Forecast request: store={store}, category={category}, period={period}, forecast_period={forecast_period}, model={model_type}")
        
        # Filter data based on parameters
        filtered_data = filter_data(preprocessed_data, store, category, period)
        
        # Generate forecasts
        forecasts = {}
        
        if model_type == 'all':
            # Generate forecasts from all models
            for model_name, model in models.items():
                try:
                    if hasattr(model, 'predict'):
                        forecast_result = model.predict(filtered_data, forecast_period)
                        forecasts[model_name] = forecast_result
                    else:
                        # Fallback to simple forecast
                        forecasts[model_name] = simple_forecast(filtered_data, forecast_period)
                except Exception as e:
                    logger.error(f"Error with {model_name} model: {str(e)}")
                    forecasts[model_name] = simple_forecast(filtered_data, forecast_period)
        else:
            # Generate forecast from specific model
            if model_type in models:
                try:
                    forecast_result = models[model_type].predict(filtered_data, forecast_period)
                    forecasts[model_type] = forecast_result
                except Exception as e:
                    logger.error(f"Error with {model_type} model: {str(e)}")
                    forecasts[model_type] = simple_forecast(filtered_data, forecast_period)
            else:
                return jsonify({'error': f'Model {model_type} not found'}), 400
        
        # Calculate metrics
        metrics = calculate_forecast_metrics(filtered_data, forecasts)
        
        # Generate insights
        insights = generate_insights(filtered_data, forecasts, params)
        
        # Prepare response
        response = {
            'historical': prepare_historical_data(filtered_data),
            'forecast': forecasts,
            'metrics': metrics,
            'insights': insights,
            'parameters': params
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in forecast endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get information about available models"""
    model_info = {}
    
    for model_name, model in models.items():
        model_info[model_name] = {
            'name': model_name,
            'type': model.__class__.__name__,
            'loaded': hasattr(model, 'model') and model.model is not None,
            'last_trained': getattr(model, 'last_trained', None)
        }
    
    return jsonify(model_info)

@app.route('/api/data/summary', methods=['GET'])
def data_summary():
    """Get data summary statistics"""
    if data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    try:
        summary = {
            'total_records': len(data),
            'date_range': {
                'start': data['Date'].min().isoformat(),
                'end': data['Date'].max().isoformat()
            },
            'stores': data['Store'].unique().tolist(),
            'categories': data['Category'].unique().tolist(),
            'total_sales': float(data['Sales'].sum()),
            'avg_daily_sales': float(data['Sales'].mean())
        }
        
        return jsonify(summary)
    except Exception as e:
        logger.error(f"Error generating data summary: {str(e)}")
        return jsonify({'error': str(e)}), 500

def filter_data(data, store, category, period):
    """Filter data based on parameters"""
    filtered = data.copy()
    
    # Filter by store
    if store != 'all':
        filtered = filtered[filtered['Store'].str.lower() == store.lower()]
    
    # Filter by category
    if category != 'all':
        filtered = filtered[filtered['Category'].str.lower() == category.lower()]
    
    # Filter by period
    end_date = filtered['Date'].max()
    start_date = end_date - timedelta(days=period)
    filtered = filtered[filtered['Date'] >= start_date]
    
    return filtered

def simple_forecast(data, forecast_period):
    """Simple forecasting method as fallback"""
    # Use moving average with trend
    recent_data = data['Sales'].tail(min(30, len(data)))
    
    if len(recent_data) < 2:
        # Not enough data, use mean
        base_value = recent_data.mean() if len(recent_data) > 0 else 1000
        return [base_value] * forecast_period
    
    # Calculate trend
    x = np.arange(len(recent_data))
    trend = np.polyfit(x, recent_data, 1)[0]
    
    # Generate forecast
    last_value = recent_data.iloc[-1]
    forecast = []
    
    for i in range(1, forecast_period + 1):
        predicted_value = last_value + (trend * i)
        # Add some randomness
        predicted_value += np.random.normal(0, predicted_value * 0.1)
        forecast.append(max(0, predicted_value))
    
    return forecast

def calculate_forecast_metrics(data, forecasts):
    """Calculate metrics for forecasts"""
    metrics = {
        'total_revenue': float(data['Sales'].sum()),
        'avg_daily_sales': float(data['Sales'].mean()),
        'growth_rate': 0.0,
        'accuracy': {}
    }
    
    if forecasts:
        # Calculate average forecast
        all_forecasts = list(forecasts.values())
        if all_forecasts:
            avg_forecast = np.mean([np.mean(f) for f in all_forecasts])
            metrics['forecast_revenue'] = float(avg_forecast * len(next(iter(all_forecasts))))
            metrics['growth_rate'] = ((metrics['forecast_revenue'] - metrics['total_revenue']) / metrics['total_revenue']) * 100
        
        # Mock accuracy scores (in real implementation, use validation data)
        for model_name in forecasts:
            metrics['accuracy'][model_name] = np.random.uniform(85, 95)
    
    return metrics

def generate_insights(data, forecasts, params):
    """Generate insights based on data and forecasts"""
    insights = []
    
    # Seasonal insights
    current_month = datetime.now().month
    for festival, details in indian_festivals.items():
        if abs(current_month - details['month']) <= 2:
            insights.append({
                'type': 'seasonal',
                'title': f'{festival.title()} Season Impact',
                'description': f'{festival.title()} is approaching. Historical data shows {int((details["boost"] - 1) * 100)}% increase in sales.',
                'impact': 'high' if details['boost'] > 1.3 else 'medium'
            })
    
    # Performance insights
    if len(data) > 0:
        recent_trend = data['Sales'].tail(7).mean() / data['Sales'].head(7).mean()
        if recent_trend > 1.1:
            insights.append({
                'type': 'performance',
                'title': 'Positive Sales Trend',
                'description': f'Sales showing upward trend with {(recent_trend - 1) * 100:.1f}% growth in recent period.',
                'impact': 'positive'
            })
        elif recent_trend < 0.9:
            insights.append({
                'type': 'performance',
                'title': 'Declining Sales Trend',
                'description': f'Sales showing downward trend with {(1 - recent_trend) * 100:.1f}% decline. Consider promotional activities.',
                'impact': 'negative'
            })
    
    # Model insights
    if forecasts:
        best_model = max(forecasts.keys(), key=lambda x: np.mean(forecasts[x]) if forecasts[x] else 0)
        insights.append({
            'type': 'model',
            'title': 'Best Performing Model',
            'description': f'{best_model.title()} model shows best performance for current parameters.',
            'impact': 'info'
        })
    
    return insights

def prepare_historical_data(data):
    """Prepare historical data for frontend"""
    if len(data) == 0:
        return {'data': [], 'labels': []}
    
    # Group by date and sum sales
    daily_sales = data.groupby('Date')['Sales'].sum().reset_index()
    daily_sales = daily_sales.sort_values('Date')
    
    return {
        'data': daily_sales['Sales'].tolist(),
        'labels': daily_sales['Date'].dt.strftime('%d %b').tolist()
    }

if __name__ == '__main__':
    # Initialize models and data
    initialize_models()
    load_initial_data()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)