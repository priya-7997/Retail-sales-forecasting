import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Indian retail specific configurations
INDIAN_FESTIVALS = {
    '2024-01-26': 'Republic Day',
    '2024-03-08': 'Holi',
    '2024-03-29': 'Ram Navami',
    '2024-04-14': 'Baisakhi',
    '2024-08-15': 'Independence Day',
    '2024-08-19': 'Raksha Bandhan',
    '2024-09-07': 'Ganesh Chaturthi',
    '2024-10-02': 'Gandhi Jayanti',
    '2024-10-12': 'Dussehra',
    '2024-11-01': 'Diwali',
    '2024-11-15': 'Bhai Dooj',
    '2024-12-25': 'Christmas'
}

MONSOON_MONTHS = [6, 7, 8, 9]  # June to September

INDIAN_STORES = {
    'mumbai': 'Mumbai - Andheri',
    'delhi': 'Delhi - Connaught Place',
    'bangalore': 'Bangalore - Koramangala',
    'chennai': 'Chennai - T.Nagar',
    'hyderabad': 'Hyderabad - Hitech City',
    'pune': 'Pune - FC Road'
}

PRODUCT_CATEGORIES = {
    'clothing': 'Clothing & Apparel',
    'electronics': 'Electronics',
    'groceries': 'Groceries & FMCG',
    'home': 'Home & Kitchen',
    'beauty': 'Beauty & Personal Care'
}

def load_data(file_path):
    """
    Load data from CSV file with error handling
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def preprocess_data(df):
    """
    Preprocess retail sales data for Indian market
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Convert date columns
    date_columns = ['Date', 'date', 'DATE']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df = df.rename(columns={col: 'date'})
            break
    
    # Handle missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Fill categorical missing values
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col != 'date':
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    # Create date features
    if 'date' in df.columns:
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    
    return df

def add_indian_features(df):
    """
    Add India-specific features to the dataset
    """
    df = df.copy()
    
    if 'date' in df.columns:
        # Festival indicators
        df['is_festival'] = df['date'].dt.strftime('%Y-%m-%d').isin(INDIAN_FESTIVALS.keys()).astype(int)
        
        # Monsoon season indicator
        df['is_monsoon'] = df['month'].isin(MONSOON_MONTHS).astype(int)
        
        # Festival season (Oct-Nov for Diwali period)
        df['is_festival_season'] = df['month'].isin([10, 11]).astype(int)
        
        # Summer sale season (Mar-May)
        df['is_summer_sale'] = df['month'].isin([3, 4, 5]).astype(int)
        
        # Wedding season (Nov-Feb)
        df['is_wedding_season'] = df['month'].isin([11, 12, 1, 2]).astype(int)
        
        # Back to school season (Jun-Jul)
        df['is_school_season'] = df['month'].isin([6, 7]).astype(int)
        
        # Pay day effect (assume 1st and 15th of month)
        df['is_payday'] = ((df['day'] == 1) | (df['day'] == 15)).astype(int)
        
        # Add lag features for sales (if sales column exists)
        sales_columns = ['sales', 'Sales', 'SALES', 'revenue', 'Revenue']
        for col in sales_columns:
            if col in df.columns:
                df[f'{col}_lag_1'] = df[col].shift(1)
                df[f'{col}_lag_7'] = df[col].shift(7)
                df[f'{col}_lag_30'] = df[col].shift(30)
                df[f'{col}_rolling_7'] = df[col].rolling(window=7).mean()
                df[f'{col}_rolling_30'] = df[col].rolling(window=30).mean()
                break
    
    return df

def create_train_test_split(df, date_column='date', test_size=0.2):
    """
    Create time-based train-test split
    """
    df = df.sort_values(date_column)
    split_index = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    
    return train_df, test_df

def calculate_metrics(y_true, y_pred):
    """
    Calculate forecasting metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }

def save_model(model, filename):
    """
    Save model to pickle file
    """
    try:
        os.makedirs('models', exist_ok=True)
        with open(f'models/{filename}', 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved as {filename}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")

def load_model(filename):
    """
    Load model from pickle file
    """
    try:
        with open(f'models/{filename}', 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filename}")
        return model
    except FileNotFoundError:
        print(f"Model file {filename} not found")
        return None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def generate_forecast_dates(start_date, periods):
    """
    Generate future dates for forecasting
    """
    start_date = pd.to_datetime(start_date)
    return pd.date_range(start=start_date, periods=periods, freq='D')

def prepare_forecast_data(last_date, periods):
    """
    Prepare data structure for future forecasting
    """
    future_dates = generate_forecast_dates(last_date + timedelta(days=1), periods)
    
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'year': future_dates.year,
        'month': future_dates.month,
        'day': future_dates.day,
        'day_of_week': future_dates.dayofweek,
        'quarter': future_dates.quarter,
        'is_weekend': future_dates.dayofweek.isin([5, 6]).astype(int),
        'is_month_start': future_dates.is_month_start.astype(int),
        'is_month_end': future_dates.is_month_end.astype(int)
    })
    
    # Add Indian-specific features
    forecast_df = add_indian_features(forecast_df)
    
    return forecast_df

def format_currency(amount):
    """
    Format currency in Indian Rupees
    """
    if amount >= 10000000:  # 1 crore
        return f"₹{amount/10000000:.1f} Cr"
    elif amount >= 100000:  # 1 lakh
        return f"₹{amount/100000:.1f} L"
    elif amount >= 1000:  # 1 thousand
        return f"₹{amount/1000:.1f} K"
    else:
        return f"₹{amount:.0f}"

def get_seasonal_multiplier(month):
    """
    Get seasonal multiplier for Indian retail
    """
    seasonal_factors = {
        1: 1.1,   # January - New Year shopping
        2: 0.9,   # February - Post-festival lull
        3: 1.0,   # March - Holi
        4: 1.0,   # April - Summer start
        5: 0.9,   # May - Hot summer
        6: 0.8,   # June - Monsoon start
        7: 0.8,   # July - Monsoon
        8: 0.9,   # August - Independence Day
        9: 1.0,   # September - Post-monsoon
        10: 1.3,  # October - Festive season
        11: 1.4,  # November - Diwali
        12: 1.2   # December - Year-end
    }
    return seasonal_factors.get(month, 1.0)

def detect_outliers(df, column, method='iqr'):
    """
    Detect outliers in sales data
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        outliers = df[z_scores > 3]
    
    return outliers

def clean_data(df):
    """
    Clean and prepare data for modeling
    """
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Remove outliers from sales data
    sales_columns = ['sales', 'Sales', 'SALES', 'revenue', 'Revenue']
    for col in sales_columns:
        if col in df.columns:
            outliers = detect_outliers(df, col)
            print(f"Detected {len(outliers)} outliers in {col}")
            # Remove extreme outliers (beyond 3 standard deviations)
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df = df[z_scores <= 3]
            break
    
    return df

def validate_data(df):
    """
    Validate data quality
    """
    validation_results = {
        'total_rows': len(df),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'date_range': None,
        'negative_sales': 0
    }
    
    # Check date range
    if 'date' in df.columns:
        validation_results['date_range'] = f"{df['date'].min()} to {df['date'].max()}"
    
    # Check for negative sales
    sales_columns = ['sales', 'Sales', 'SALES', 'revenue', 'Revenue']
    for col in sales_columns:
        if col in df.columns:
            validation_results['negative_sales'] = (df[col] < 0).sum()
            break
    
    return validation_results

def get_model_config():
    """
    Get default model configurations
    """
    return {
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        },
        'prophet': {
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'interval_width': 0.95,
            'changepoint_prior_scale': 0.05
        },
        'arima': {
            'order': (2, 1, 2),
            'seasonal_order': (1, 1, 1, 12),
            'trend': 'c'
        }
    }

# Example usage and testing
if __name__ == "__main__":
    # Test data loading
    print("Testing utils.py functions...")
    
    # Test festival dates
    print("Indian Festivals:")
    for date, festival in INDIAN_FESTIVALS.items():
        print(f"  {date}: {festival}")
    
    # Test currency formatting
    print("\nCurrency Formatting:")
    print(f"  1000: {format_currency(1000)}")
    print(f"  150000: {format_currency(150000)}")
    print(f"  12500000: {format_currency(12500000)}")
    
    # Test seasonal multipliers
    print("\nSeasonal Multipliers:")
    for month in range(1, 13):
        print(f"  Month {month}: {get_seasonal_multiplier(month)}")
    
    print("\nUtils.py loaded successfully!")