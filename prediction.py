import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Load product data
product_data = pd.read_csv("competitor_data.csv")
product_data.head()

# Preprocessing data
product_data["Date"] = pd.to_datetime(product_data['Date'], errors="coerce")
product_data = product_data.dropna(subset=["Date"])  # Remove missing Date values
product_data.set_index("Date", inplace=True)
product_data = product_data.sort_index()

# Convert Discount column to numeric and handle errors
product_data["Discount"] = pd.to_numeric(product_data["Discount"], errors="coerce")
product_data = product_data.dropna(subset=["Discount"])  # Remove rows with missing Discount values
product_data = product_data.sort_index()

# Extract the 'Discount' series for analysis
discounts = product_data["Discount"]

# Set ARIMA parameters
p = 5  # Number of past observations (lags)
q = 0  # Number of error terms to include
d = 1  # Degree of differencing to make the data stationary

# Fit ARIMA model
model = ARIMA(discounts, order=(p, d, q))
model_fit = model.fit()

# Forecast the next 'days' future values
days = 5
forecast = model_fit.forecast(steps=days)

# Generate future dates for the forecasted values
future_dates = pd.date_range(start=discounts.index[-1] + pd.Timedelta(days=days), periods=days)
forecasted_df = pd.DataFrame({"Date": future_dates, "Predicted Discounts": forecast})

# Display the forecasted values
forecasted_df.head()