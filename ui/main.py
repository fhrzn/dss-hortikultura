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

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(name)s - %(message)s")
logger = logging.getLogger(__name__)


dbutils.init_db()
logging.info(st.session_state)


st.write("# TODO: DSS Main Application Here")