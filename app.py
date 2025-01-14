import streamlit as st
import joblib

model = joblib.load('spam_filter_model.pkl')

#styling!
st.markdown(
    """
    <style>
    .stApp {
        background-color: #A1D6E2; 
        color: #F1F1F2;
    }
    .stTextArea label {
        color: #F1F1F2 !important; 
        font-weight: bold;
    }
    .stButton > button {
        background-color: #1995AD; 
        color: white; 
    }
    .stSuccess {
        background-color: #98ff98 !important; 
        color: #004d40 !important; 
        border-radius: 5px;
        padding: 10px;
    }
    .stError {
        background-color: #ff6b6b !important; 
        color: white !important; 
        border-radius: 5px;
        padding: 10px;
    }
    .stWarning {
        background-color: #ffcc00 !important; 
        color: black !important; 
        border-radius: 5px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("👀 SPAM ah!?")

email_content = st.text_area("Enter the email's content", placeholder="Type or paste email content here...")

if st.button("Detect Spam"):
    if email_content.strip():  #check for user input
        result = model.predict([email_content])  #apply the ML magic ^-^
        if result[0] == 1:  
            st.markdown('<div class="stError">🚨 This email is SPAM!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="stSuccess">✅ This email is NOT spam.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="stWarning">⚠️ Please enter email content!</div>', unsafe_allow_html=True)

#let them know how accurate the models prediction is ~ ~
st.markdown("""---""") 
st.metric(label="Model Accuracy", value="97.5%", delta=None,)
