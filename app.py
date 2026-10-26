
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
    model="qwen/qwen3-8b",
    temperature=0,
    max_tokens=700
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

    prompt = """
    Analyze this stock chart briefly.

    Include:
    1. Overall trend
    2. Support and resistance
    3. Moving averages
    4. RSI or MACD, if visible
    5. Bullish or bearish signals
    6. Risks and uncertainty

    Use short bullet points.
    Do not invent values that are not visible.
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
