
import streamlit as st
import os
import base64
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

st.set_page_config(
    page_title="Stock Chart Analyst",
    page_icon="📈"
)

st.title("📈 Stock Market Chart Analysis Agent")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("Missing GROQ_API_KEY. Check your .env file.")
    st.stop()

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    api_key=api_key,
    temperature=0
)

image = st.file_uploader(
    "Upload a stock chart",
    type=["png", "jpg", "jpeg"]
)

question = st.text_input(
    "Ask a question",
    "Analyze the trend, pattern, support, resistance, rally and volume."
)

if image and st.button("Analyze Chart"):
    st.image(
        image,
        caption="Uploaded Stock Chart",
        use_container_width=True
    )

    encoded = base64.b64encode(
        image.getvalue()
    ).decode("utf-8")

    mime = image.type or "image/png"

    prompt = f"""
    Analyze the uploaded stock chart in plain language.

    User question: {question}

    Explain:
    1. Overall price trend
    2. Visible chart patterns
    3. Approximate support and resistance
    4. Rally period and percentage, if readable
    5. Unusually high trading volume
    6. Relationship between price and volume
    7. Risks and uncertainty

    Do not invent unreadable dates or price levels.
    Separate observations from predictions.
    This is not investment advice.
    """

    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime};base64,{encoded}"
            }
        }
    ])

    with st.spinner("Analyzing chart..."):
        try:
            result = llm.invoke([message])
            st.subheader("Analysis Result")
            st.markdown(result.content)

        except Exception as e:
            st.error(f"Analysis failed: {e}")
