
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
st.caption("Upload a stock chart to analyze its visible patterns.")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("Missing GROQ_API_KEY. Check your local .env file.")
    st.stop()

# Vision-capable Groq model
llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    groq_api_key=api_key,
    temperature=0,
    max_tokens=500
)

image = st.file_uploader(
    "Upload a stock chart",
    type=["png", "jpg", "jpeg"]
)

question = st.text_input(
    "Ask a question",
    "Analyze the trend, pattern, support, resistance, rally and volume."
)

if image:
    st.image(
        image,
        caption="Uploaded Stock Chart",
        use_container_width=True
    )

if image and st.button("Analyze Chart", type="primary"):

    if image.size > 20 * 1024 * 1024:
        st.error("Image is too large. Please upload an image under 20 MB.")
        st.stop()

    encoded = base64.b64encode(
        image.getvalue()
    ).decode("utf-8")

    mime = image.type or "image/png"

    prompt = f"""
    You are a stock chart analysis assistant.

    Analyze only the information visible in this image.

    User question: {question}

    Give a concise response with:
    1. Overall trend
    2. Chart pattern
    3. Support and resistance, if identifiable
    4. Moving averages and volume, if visible
    5. RSI or MACD, only if displayed
    6. Possible bullish or bearish signals
    7. Risks and uncertainty

    Use short bullet points.
    Do not invent prices, indicator values, or trading signals.
    If something is not visible, clearly say so.
    This is educational analysis, not financial advice.
    """

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{encoded}"
                }
            }
        ]
    )

    with st.spinner("Analyzing your stock chart..."):
        try:
            result = llm.invoke([message])

            st.subheader("📊 Analysis Result")
            st.markdown(result.content)

        except Exception as e:
            error = str(e)

            if "429" in error or "rate_limit" in error:
                st.error(
                    "Groq rate limit reached. Wait for your quota "
                    "to reset, then try again. Check your Groq "
                    "Console for current limits."
                )

            elif "404" in error or "model_not_found" in error:
                st.error(
                    "The model is unavailable for your account. "
                    "Check the model ID and access in the Groq Console."
                )

            elif "401" in error or "invalid_api_key" in error:
                st.error(
                    "Invalid API key. Create a valid Groq key and "
                    "update your local .env file."
                )

            else:
                st.error(f"Analysis failed: {error}")
