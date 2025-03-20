
import streamlit as st
import requests
import json

# Groq API Details
API_URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {
    "Authorization": "Bearer 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",
    "Content-Type": "application/json"
}

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a health assistant that provides medical guidance based on symptoms. Do not give prescriptions. Recommend seeing a doctor when necessary."}
    ]

# Streamlit UI
st.title("🩺 AI Health Assistant")
st.write("Enter your symptoms, and I'll help you understand possible conditions. **Note:** This is not medical advice.")

# User input
user_prompt = st.text_input("Enter your symptoms (e.g., fever, cough, headache):")

if st.button("Ask AI"):
    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        data = {
            "model": "qwen-2.5-32b",
            "messages": st.session_state.messages,
            "temperature": 0.5
        }

        try:
            response = requests.post(API_URL, data=json.dumps(data), headers=HEADERS)
            response_json = response.json()

            # Extract chatbot response
            if "choices" in response_json and len(response_json["choices"]) > 0:
                bot_reply = response_json["choices"][0]["message"]["content"]
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                st.success(bot_reply)
            else:
                st.error("Sorry, I couldn't process that.")
        except Exception as e:
            st.error(f"Error: {e}")

# Show chat history
st.subheader("Chat History:")
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.write(f"👤 **You:** {msg['content']}")
    elif msg["role"] == "assistant":
        st.write(f"🤖 **Bot:** {msg['content']}")

