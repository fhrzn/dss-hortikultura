import streamlit as st
from controller import tanaman
import logging
from utils import dbutils

logger = logging.getLogger(__name__)


dbutils.init_db()
logging.info(st.session_state)


st.set_page_config(
    page_title="Basis Data Hortikultura DSS",
    page_icon="🌱",
    initial_sidebar_state="collapsed",
)


st.write("# Basis Data")


tab1, tab2 = st.tabs(["🌱 Profil Ideal Tanaman", "🪨 Profil Kondisi Lahan"])

with tab1:
    # CREATE
    with st.expander("Form input", expanded=False):
        with st.form("form", border=False, clear_on_submit=True):
            # create input fields
            nama_tanaman = st.text_input("Nama Tanaman", placeholder="ex: Bawang Merah")
            st.write("**Suhu**")
            suhu_ideal = st.text_input("Suhu Ideal")
            suhu_batas_interpolasi = st.text_input("Batas Interpolasi Suhu")
            st.write("**Curah Hujan**")
            curah_hujan_ideal = st.text_input("Curah Hujan Ideal")
            curah_hujan_batas_interpolasi = st.text_input("Batas Interpolasi Curah Hujan")
            st.write("**Kelembaban**")
            kelembaban_ideal = st.text_input("Kelembaban Ideal")
            kelembaban_batas_interpolasi = st.text_input("Batas Interpolasi Kelembaban")
            st.write("**pH**")
            ph_ideal = st.text_input("pH Ideal")
            ph_batas_interpolasi = st.text_input("Batas Interpolasi pH")
            st.write("**Kemiringan**")
            kemiringan_ideal = st.text_input("Kemiringan Ideal")
            kemiringan_batas_interpolasi = st.text_input("Batas Interpolasi Kemiringan")
            st.write("**Topografi**")
            topografi_ideal = st.text_input("Topografi Ideal")
            topografi_batas_interpolasi = st.text_input("Batas Interpolasi Topografi")

            # submit btn
            submit = st.form_submit_button("Submit Data")
            if submit:
                tanaman.insert_tanaman(
                    db=st.session_state["db"],
                    nama_tanaman=nama_tanaman,
                    suhu_ideal=suhu_ideal,
                    suhu_batas_interpolasi=suhu_batas_interpolasi,
                    curah_hujan_ideal=curah_hujan_ideal,
                    curah_hujan_batas_interpolasi=curah_hujan_batas_interpolasi,
                    kelembaban_ideal=kelembaban_ideal,
                    kelembaban_batas_interpolasi=kelembaban_batas_interpolasi,
                    ph_ideal=ph_ideal,
                    ph_batas_interpolasi=ph_batas_interpolasi,
                    kemiringan_ideal=kemiringan_ideal,
                    kemiringan_batas_interpolasi=kemiringan_batas_interpolasi,
                    topografi_ideal=topografi_ideal,
                    topografi_batas_interpolasi=topografi_batas_interpolasi
                )

    # READ
    with st.expander("Data Tanaman", expanded=True):
        st.empty()
        tanamans = tanaman.read_tanaman(db=st.session_state["db"])
        st.dataframe(tanamans)
        


with tab2:
    # TODO: CRUD profil kondisi lahan
    pass

    