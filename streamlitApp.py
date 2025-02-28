import streamlit as st
import pandas as pd
import random

# Set page title
st.title("Interactive Streamlit App")

# Text input for user name
name = st.text_input("Enter your name:")

# Number counter
count = st.number_input("Enter a number:", min_value=1, max_value=100, value=10)

# Button to show message
if st.button("Greet Me"):
    st.success(f"Hello, {name}! You chose {count}.")

# Display a sample data table
st.subheader("Sample Data Table")
data = {"Name": ["chinni", "nani", "pandu"], "Age": [20, 25, 22], "Score": [95, 90, 88]}
df = pd.DataFrame(data)
st.table(df)

# Random quote generator
quotes = [
    "Believe in yourself!",
    "Stay positive, work hard, make it happen.",
    "Success is not final, failure is not fatal.",
    "Do something today that your future self will thank you for."
]
if st.button("Get a Motivational Quote"):
    st.write(random.choice(quotes))

# Sidebar with a selection box
st.sidebar.title("Choose an Option")
option = st.sidebar.radio("Select one:", ["Home", "About", "Contact"])
st.sidebar.write(f"You selected: {option}")
