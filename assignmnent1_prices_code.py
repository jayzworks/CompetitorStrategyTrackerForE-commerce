import json
from datetime import datetime
import pandas as pd

# Mock JSON responses for 5 products from different websites
json_responses = [
    """{
        "pid": "MOBGNE4SXH2BDXTV",
        "history": {
            "1679341044": 30990.0,
            "1679427444": 30990.0,
            "1679513844": 30990.0,
            "1679600244": 30990.0,
            "1679686644": 30990.0
        },
        "price_fetched_at": "2025-03-16T02:18:30.032628Z",
        "lowest_price": 22999.0,
        "highest_price": 35499.0,
        "average_price": 29586.0,
        "drop_chances": 7.0,
        "rating": 0.0,
        "rating_count": 0,
        "stock": false,
        "price": 27499.0,
        "url": "https://www.flipkart.com/samsung-galaxy-a34-5g-awesome-silver-128-gb/p/itm2dd3e0ff525a8",
        "name": "SAMSUNG Galaxy A34 5G (Awesome Silver, 128 GB)",
        "reviews": [
            {"text": "Very nice mobile phone from Samsung. Really it's superb", "rating": 5},
            {"text": "Worth it.", "rating": 4}
        ]
    }""",
    """{
        "pid": "MOBGHWFHUYWGB5F2",
        "history": {
            "1661785816": 89900.0,
            "1661872216": 89900.0
        },
        "price_fetched_at": "2025-03-17T11:32:16.550318Z",
        "lowest_price": 54999.0,
        "highest_price": 89900.0,
        "average_price": 78621.0,
        "drop_chances": 9.0,
        "rating": 0.0,
        "rating_count": 0,
        "stock": true,
        "price": 69900.0,
        "url": "https://www.flipkart.com/apple-iphone-14-plus-blue-128-gb/p/itmac8385391b02b",
        "name": "Apple iPhone 14 Plus (Blue, 128 GB)",
        "reviews": [
            {"text": "Camera is too cool", "rating": 5},
            {"text": "Best in class", "rating": 4}
        ]
    }""",
    """{
        "pid": "MOBGH2UVMHEPGSHM",
        "history": {
            "1672950992": 24999.0,
            "1673033218": 24999.0
        },
        "price_fetched_at": "2025-03-14T05:49:48.032956Z",
        "lowest_price": 15999.0,
        "highest_price": 27999.0,
        "average_price": 24289.0,
        "drop_chances": 1.0,
        "rating": 0.0,
        "rating_count": 0,
        "stock": true,
        "price": 18602.0,
        "url": "https://www.flipkart.com/redmi-note-12-pro-5g-onyx-black-128-gb/p/itmbc9fd7adaa32a",
        "name": "Redmi Note 12 Pro",
        "reviews": [
            {"text": "Very comfortable and stylish.", "rating": 5},
            {"text": "Best phone.. Picture quality is awesome....", "rating": 2}
        ]
    }""",
    """{
        "pid": "B09XS7JWHH",
        "history": {
            "1677648877": 33990.0,
            "1683504681": 26806.0
        },
        "price_fetched_at": "2025-03-17T10:34:57.568059Z",
        "lowest_price": 21990.0,
        "highest_price": 34990.0,
        "average_price": 28401.0,
        "drop_chances": 63.0,
        "rating": 4.5,
        "rating_count": 14951,
        "stock": true,
        "price": 29990.0,
        "url": "https://www.amazon.in/dp/B09XS7JWHH",
        "name": "Sony WH-1000XM5 Wireless Noise Canceling Headphones",
        "reviews": [
            {"text": "Good Product and Great Battery Life..", "rating": 5}
        ]
    }"""
]

# Function to process a single product's JSON data
def process_product(json_response):
    data = json.loads(json_response)  # Convert JSON string to dictionary
    
    # Extract relevant data
    product_name = data.get("name", "Unknown Product")
    source = data.get("url", "Unknown Source")
    price_history = data.get("history", {})
    current_price = data.get("price", 0)
    highest_price = data.get("highest_price", 0)
    reviews = data.get("reviews", [])
    
    # Calculate discount percentage
    discount = 0
    if highest_price > 0:
        discount = round(((highest_price - current_price) / highest_price) * 100, 2)
    
    # Process price history
    history_data = []
    for timestamp, price in price_history.items():
        try:
            date = datetime.fromtimestamp(int(timestamp))  # Convert Unix timestamp to date
            formatted_date = date.strftime("%Y-%m-%d")     # Format date as YYYY-MM-DD
        except ValueError:
            formatted_date = "Invalid Date"
        
        history_data.append({
            "Date": formatted_date,
            "Price": price,
            "Product Name": product_name,
            "Discount (%)": discount,
            "Source": source
        })
    
    # Process reviews
    review_data = []
    for review in reviews:
        review_data.append({
            "Product Name": product_name,
            "Review Text": review.get("text", "No review text"),
            "Rating": review.get("rating", 0),
            "Source": source
        })
    
    return history_data, review_data

# Process all products
all_price_data = []
all_review_data = []
for json_response in json_responses:
    price_data, review_data = process_product(json_response)
    all_price_data.extend(price_data)
    all_review_data.extend(review_data)

# Convert to DataFrames
price_df = pd.DataFrame(all_price_data)
review_df = pd.DataFrame(all_review_data)

# Save to CSV files
price_df.to_csv("price_history_updated.csv", index=False)
review_df.to_csv("reviews_updated.csv", index=False)

print("price_history_updated.csv generated successfully.")
print("reviews_updated.csv generated successfully.")
