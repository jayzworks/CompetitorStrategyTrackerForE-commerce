import json
from datetime import datetime

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from transformers import pipeline

API_KEY = "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"  # Groq API Key

def truncate_text(text, max_length=512):
    return text[:max_length]


def load_competitor_data():
    """Load competitor data from a CSV file."""
    data = pd.read_csv("competitor_price_data.csv")
    print(data.head())
    return data


def load_reviews_data():
    """Load reviews data from a CSV file."""
    reviews = pd.read_csv("reviews_data.csv")
    return reviews


def analyze_sentiment(reviews):
    """Analyze customer sentiment for reviews."""
    sentiment_pipeline = pipeline("sentiment-analysis")
    return sentiment_pipeline(reviews)


def train_predictive_model(data):
    """Train a predictive model for competitor pricing strategy."""
    data["Discount"] = data["Discount"].str.replace("%", "").astype(float)
    data["Price"] = data["Price"].astype(int).round(2)
    data["Predicted_Discount"] = data["Discount"] + (data["Price"] * 0.05).round(2)

    X = data[["Price", "Discount"]]
    y = data["Predicted_Discount"]
    print(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, train_size=0.8
    )

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model


import numpy as np
import pandas as pd


def forecast_discounts_arima(data, future_days=7):
    """
    Forecast future discounts using ARIMA.
    :param data: DataFrame containing historical discount data (with a datetime index).
    :param future_days: Number of days to forecast.
    :return: DataFrame with historical and forecasted discounts.
    """

    if data.empty:
        st.warning("No valid discount data available for forecasting.")
        return pd.DataFrame()  # Return an empty DataFrame

    discount_series = data["Discount"]

    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            raise ValueError("Index must be datetime or convertible to datetime.") from e

    # Check if discount_series is empty after processing
    if discount_series.empty:
        st.warning("No valid historical discount data for ARIMA model.")
        return pd.DataFrame()

    model = ARIMA(discount_series, order=(0, 1, 2))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=future_days)
    future_dates = pd.date_range(
        start=pd.Timestamp.today().normalize(),  # Start from today
        periods=future_days
    )

    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Discount": forecast.round(2)})
    forecast_df.set_index("Date", inplace=True)

    return forecast_df


def generate_strategy_recommendation(product_name, competitor_data, sentiment):
    """Generate strategic recommendations using an LLM."""
    date = datetime.now()
    prompt = f"""
    You are a highly skilled business strategist specializing in e-commerce. Based on the following details, suggest actionable strategies to optimize pricing, promotions, and customer satisfaction for the selected product:

1. **Product Name**: {product_name}

2. **Competitor Data** (including current prices, discounts, and predicted discounts):
{competitor_data}

3. **Sentiment Analysis**:
{sentiment}


5. **Today's Date**: {str(date)}

### Task:
- Analyze the competitor data and identify key pricing trends.
- Leverage sentiment analysis insights to highlight areas where customer satisfaction can be improved.
- Use the discount predictions to suggest how pricing strategies can be optimized over the next 5 days.
- Recommend promotional campaigns or marketing strategies that align with customer sentiments and competitive trends.
- Ensure the strategies are actionable, realistic, and geared toward increasing customer satisfaction, driving sales, and outperforming competitors.

Provide your recommendations in a structured format:
1. **💰Pricing Strategy**
2. **🎯Promotional Campaign Ideas**
3. **⭐Customer Satisfaction Recommendations**
    """

    messages = [{"role": "user", "content": prompt}]

    data = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "llama3-8b-8192",
        "temperature": 0,
    }

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}

    res = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(data),
        headers=headers,
        timeout=10,
    )
    res = res.json()
    response = res["choices"][0]["message"]["content"]
    return response


####-----------------------Frontend---------------------------##########

st.set_page_config(page_title="E-Commerce Competitor Strategy Dashboard", layout= "wide")


st.title("🚀 E-Commerce Competitor Strategy Dashboard")
st.sidebar.header("Select a Product")

competitor_data = load_competitor_data()
reviews_data = load_reviews_data()

products = competitor_data['Product name'].unique()

selected_product = st.sidebar.selectbox("🔍Choose a product to analyze:", products)

product_data = competitor_data[competitor_data["Product name"] == selected_product]
product_reviews = reviews_data[reviews_data["Product name"] == selected_product]

st.markdown(f"<h2 style='color: black;'>🛒 Competitor Analysis for {selected_product}</h2>", unsafe_allow_html=True)
st.subheader("💲Competitor Price Data")
st.table(product_data.tail(5).round(2).set_index(product_data.columns[0]))


# Add Expander for Price & Discount Trends
product_data["Date"] = pd.to_datetime(product_data["Date"])

with st.expander("🔍 View Competitor Price and Discount Trends"):
    st.subheader("📉 Price & Discount Trends for Flipkart & Amazon")

    # ✅ Create columns for side-by-side graphs
    col1, col2 = st.columns(2)

    # ✅ Price Trend Graph (Left)
    with col1:
        st.markdown("### 📊 Price Trend")
        fig_price = px.line(
            product_data, 
            x="Date",  # ✅ Use full date
            y="Price", 
            color="Source",  
            title="Price Trends - Flipkart vs Amazon",
            markers=True, 
            line_shape="linear",
            color_discrete_map={"Flipkart": "blue", "Amazon": "orange"}
        )
        fig_price.update_xaxes(
            tickformat="%Y-%m-%d",  # ✅ Force full date format
            tickangle=45  # Rotate for better readability
        )
        st.plotly_chart(fig_price, use_container_width=True)

    # ✅ Discount Trend Graph (Right)
    with col2:
        st.markdown("### 📊 Discount Trend")
        fig_discount = px.line(
            product_data, 
            x="Date",  # ✅ Use full date
            y="Discount", 
            color="Source",  
            title="Discount Trends - Flipkart vs Amazon",
            markers=True, 
            line_shape="linear",
            color_discrete_map={"Flipkart": "blue", "Amazon": "orange"}
        )
        fig_discount.update_xaxes(
            tickformat="%Y-%m-%d",  # ✅ Force full date format
            tickangle=45  # Rotate for better readability
        )
        st.plotly_chart(fig_discount, use_container_width=True)
if not product_reviews.empty:
    product_reviews["reviews"] = product_reviews["reviews"].apply(
        lambda x: truncate_text(x, 512)
    )
    reviews = product_reviews["reviews"].tolist()
    sentiments = analyze_sentiment(reviews)

    st.subheader("😊 Customer Sentiment Analysis")
    
    # Convert sentiment results into a DataFrame
    sentiment_df = pd.DataFrame(sentiments)
    
    # Define colors for sentiment categories
    color_map = {"POSITIVE": "green", "NEGATIVE": "red"}  
    
    # Create bar chart with custom colors
    fig = px.bar(
        sentiment_df, 
        x="label", 
        y="score",  # Ensure y-axis represents the sentiment score
        title="Sentiment Analysis Results",
        color="label",
        color_discrete_map=color_map
    )
    
    st.plotly_chart(fig)
else:
    st.write("No reviews available for this product.")

# Preprocessing

product_data["Date"] = pd.to_datetime(product_data["Date"], errors="coerce")
product_data = product_data.dropna(subset=["Date"])
product_data.set_index("Date", inplace=True)
product_data = product_data.sort_index()

product_data["Discount"] = pd.to_numeric(product_data["Discount"], errors="coerce")
product_data = product_data.dropna(subset=["Discount"])

# Forecasting Model
product_data_with_predictions = forecast_discounts_arima(product_data)


st.subheader("🏷️Competitor Current and Predicted Discounts")
st.table(product_data_with_predictions.style.format({"Predicted_Discount": "{:.2f}"}))
recommendations = generate_strategy_recommendation(
    selected_product,
    product_data_with_predictions,
    sentiments if not product_reviews.empty else "No reviews available",
)

st.subheader("📢 Strategic Recommendations")
st.write(recommendations)


