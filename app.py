import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import os

from pandasai import SmartDataframe
from pandasai.llm.azure_openai import AzureOpenAI
from pandasai.responses.response_parser import ResponseParser

# ----------------------------
# Custom Response Parser for Streamlit
# ----------------------------
class StreamlitFormatter(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)


    def format_dataframe(self, result):
        st.subheader("📊 Resulting DataFrame")
        st.dataframe(result["value"])
    def format_plot(self, result):
        st.subheader("📈 Generated Plot")
        st.image(result["value"])
    def format_other(self, result):
        st.subheader("📝 Response")
        st.write(result["value"])
    


# ----------------------------
# Load Environment Variables
# ----------------------------
load_dotenv()

# ----------------------------
# Streamlit Page Configuration
# ----------------------------
st.set_page_config(page_title="📊 Data Analyst Assistant", layout="wide")
st.title("🤖 AI-Powered Data Analyst Assistant")
st.markdown("Upload your Excel file and interact with your data using natural language prompts.")

# ----------------------------
# File Upload Section
# ----------------------------
uploaded_file = st.file_uploader("📁 Upload Excel File", type=["xlsx"])

dataframe = None
if uploaded_file is not None:
    try:
        dataframe = pd.read_excel(uploaded_file)
        dataframe.columns = dataframe.columns.str.upper()
        st.success("✅ Data Uploaded Successfully")
        st.subheader("Preview of Uploaded Data")
        st.dataframe(dataframe.head())
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")

st.markdown("---")

# ----------------------------
# Prompt Input Section
# ----------------------------
st.subheader("🧠 Ask Your Data a Question")
user_prompt = st.text_area("💬 Enter your prompt (e.g., 'Show me a plot of sales by region')")

# ----------------------------
# Azure OpenAI LLM Setup
# ----------------------------
llm = AzureOpenAI(
    api_token=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    model=os.getenv("AZURE_OPENAI_MODEL"),
    api_version=os.getenv("AZURE_OPENAI_VERSION")
)

# ----------------------------
# Handle Submission
# ----------------------------
if st.button("🚀 Analyze"):
    if dataframe is None:
        st.warning("📂 Please upload a file before submitting a prompt.")
    elif not user_prompt.strip():
        st.warning("✏️ Please enter a prompt.")
    else:
        with st.spinner("⏳ Processing your request..."):
            smart_df = SmartDataframe(dataframe, config={
                "llm": llm,
                "response_parser": StreamlitFormatter
            })
            st.write(smart_df.chat(user_prompt))
            