import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os

# Load insider trading data
insider_data = pd.read_csv('insider_trading_data.csv')

# Load stock price data
stock_price_data = {}
for file in os.listdir('stock_price_folder'):
    if file.endswith('.csv'):
        symbol = file.split('.')[0]
        stock_price_data[symbol] = pd.read_csv(os.path.join('stock_price_folder', file))


# Function to calculate price change
def calculate_price_change(symbol, date, days):
    if symbol not in stock_price_data:
        return np.nan

    stock_df = stock_price_data[symbol]
    date_index = stock_df[stock_df['Date'] == date].index

    if len(date_index) == 0:
        return np.nan

    current_index = date_index[0]
    previous_index = max(0, current_index - days)

    current_price = stock_df.loc[current_index, 'Close']
    previous_price = stock_df.loc[previous_index, 'Close']

    return (current_price - previous_price) / previous_price


# Feature engineering
def engineer_features(df):
    df['TRANSACTION_VALUE'] = df['NO. OF SECURITIES (ACQUIRED/DISPLOSED)'] * df[
        'VALUE OF SECURITY (ACQUIRED/DISPLOSED)']
    df['DAYS_TO_INTIMATION'] = (pd.to_datetime(df['DATE OF INITMATION TO COMPANY']) - pd.to_datetime(
        df['DATE OF ALLOTMENT/ACQUISITION TO'])).dt.days
    df['PRICE_CHANGE_5D'] = df.apply(
        lambda row: calculate_price_change(row['SYMBOL'], row['DATE OF ALLOTMENT/ACQUISITION TO'], 5), axis=1)
    df['PRICE_CHANGE_30D'] = df.apply(
        lambda row: calculate_price_change(row['SYMBOL'], row['DATE OF ALLOTMENT/ACQUISITION TO'], 30), axis=1)

    return df


# Preprocess data
def preprocess_data(df):
    # Convert categorical variables to numerical
    df = pd.get_dummies(df, columns=['REGULATION', 'CATEGORY OF PERSON', 'TYPE OF SECURITY (ACQUIRED/DISPLOSED)',
                                     'ACQUISITION/DISPOSAL TRANSACTION TYPE', 'MODE OF ACQUISITION'])

    # Handle missing values
    df = df.fillna(0)

    return df


# Main process
insider_data = engineer_features(insider_data)
insider_data = preprocess_data(insider_data)

# Define features and target
features = ['TRANSACTION_VALUE', 'DAYS_TO_INTIMATION', 'PRICE_CHANGE_5D', 'PRICE_CHANGE_30D',
            'NO. OF SECURITY (PRIOR)', '% SHAREHOLDING (PRIOR)', 'NO. OF SECURITIES (ACQUIRED/DISPLOSED)',
            'NO. OF SECURITY (POST)'] + [col for col in insider_data.columns if col.startswith(('REGULATION_',
                                                                                                'CATEGORY OF PERSON_',
                                                                                                'TYPE OF SECURITY (ACQUIRED/DISPLOSED)_',
                                                                                                'ACQUISITION/DISPOSAL TRANSACTION TYPE_',
                                                                                                'MODE OF ACQUISITION_'))]

X = insider_data[features]
y = insider_data['SUSPICIOUS_FLAG']  # You need to define this column based on your criteria for suspicious trades

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
print(classification_report(y_test, y_pred))


# Function to flag future suspicious trades
def flag_suspicious_trade(trade_data):
    trade_data = engineer_features(trade_data)
    trade_data = preprocess_data(trade_data)
    trade_features = trade_data[features]
    trade_features_scaled = scaler.transform(trade_features)
    prediction = model.predict(trade_features_scaled)
    return prediction[0]


# Example usage
new_trade = pd.DataFrame({
    'SYMBOL': ['ABC'],
    'COMPANY': ['ABC Company'],
    'REGULATION': ['SAST'],
    'NAME OF THE ACQUIRER/DISPOSER': ['John Doe'],
    'CATEGORY OF PERSON': ['Promoter'],
    'TYPE OF SECURITY (PRIOR)': ['Equity Shares'],
    'NO. OF SECURITY (PRIOR)': [1000],
    '% SHAREHOLDING (PRIOR)': [0.1],
    'TYPE OF SECURITY (ACQUIRED/DISPLOSED)': ['Equity Shares'],
    'NO. OF SECURITIES (ACQUIRED/DISPLOSED)': [500],
    'VALUE OF SECURITY (ACQUIRED/DISPLOSED)': [100],
    'ACQUISITION/DISPOSAL TRANSACTION TYPE': ['Market Purchase'],
    'TYPE OF SECURITY (POST)': ['Equity Shares'],
    'NO. OF SECURITY (POST)': [1500],
    '% POST': [0.15],
    'DATE OF ALLOTMENT/ACQUISITION FROM': ['2023-01-01'],
    'DATE OF ALLOTMENT/ACQUISITION TO': ['2023-01-01'],
    'DATE OF INITMATION TO COMPANY': ['2023-01-05'],
    'MODE OF ACQUISITION': ['Market'],
    'DERIVATIVE TYPE SECURITY': ['N/A'],
    'DERIVATIVE CONTRACT SPECIFICATION': ['N/A'],
    'NOTIONAL VALUE(BUY)': [0],
    'NUMBER OF UNITS/CONTRACT LOT SIZE (BUY)': [0],
    'NOTIONAL VALUE(SELL)': [0],
    'NUMBER OF UNITS/CONTRACT LOT SIZE  (SELL)': [0],
    'EXCHANGE': ['NSE'],
    'REMARK': [''],
    'BROADCASTE DATE AND TIME': ['2023-01-06 09:00:00'],
    'XBRL': ['Yes']
})

is_suspicious = flag_suspicious_trade(new_trade)
print(f"Is the trade suspicious? {'Yes' if is_suspicious else 'No'}")