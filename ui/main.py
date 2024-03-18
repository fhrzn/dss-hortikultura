import streamlit as st
import logging
from utils import dbutils
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Hortikultura Decision Support System",
    page_icon="🌱",
    initial_sidebar_state="collapsed",
)

logger = logging.getLogger(__name__)


dbutils.init_db()
logging.info(st.session_state)


st.write("# TODO: DSS Main Application Here")