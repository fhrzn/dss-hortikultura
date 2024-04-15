import streamlit as st
from controller import tanaman, lahan
import logging
from utils import dbutils
from sqlalchemy.orm import Session

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
    with st.expander("Input Tanaman Baru", expanded=False):
        with st.form("form", border=False, clear_on_submit=True):
            # create input fields
            jenis_tanaman = st.text_input("Jenis Tanaman", placeholder="ex: Bawang Merah")
            st.write("**Suhu**")
            suhu = st.text_input("Suhu Ideal", placeholder="ex: 10-30")
            suhu_interpolasi = st.text_input("Batas Interpolasi Suhu", placeholder="ex: 0,25,32,57")
            st.write("**Curah Hujan**")
            curah_hujan = st.text_input("Curah Hujan Ideal", placeholder="ex: 350-800")
            curah_hujan_interpolasi = st.text_input("Batas Interpolasi Curah Hujan", placeholder="ex: 0,300,2500,2800")
            st.write("**Kelembapan**")
            kelembapan = st.text_input("Kelembapan Ideal", placeholder="ex: 80-90")
            kelembapan_interpolasi = st.text_input("Batas Interpolasi Kelembapan", placeholder="ex: 0,80,90,170")
            st.write("**Jenis Tanah**")
            jenis_tanah = st.multiselect("Jenis Tanah", ["mediteran", "regosol", "latosol", "grumosol", "aluvial"])
            st.write("**Tekstur Tanah**")
            tekstur_tanah = st.multiselect("Tekstur Tanah", ["berdebu", "liat", "liat berpasir", "liat berdebu" "lempung berpasir", "lempung berdebu"])
            st.write("**pH**")
            ph = st.text_input("pH Ideal", placeholder="ex: 6-8")
            ph_interpolasi = st.text_input("Batas Interpolasi pH", placeholder="ex: 0,5.6,6.5,12.1")
            st.write("**Kemiringan**")
            kemiringan = st.text_input("Kemiringan Ideal", placeholder="ex: 5.5-16")
            kemiringan_interpolasi = st.text_input("Batas Interpolasi Kemiringan", placeholder="ex: 0,30,60")
            st.write("**Topografi**")
            topografi = st.text_input("Topografi Ideal", placeholder="ex: 700-1000")
            topografi_interpolasi = st.text_input("Batas Interpolasi Topografi", placeholder="ex: 0,700,1000,1700")

            # submit btn
            submit = st.form_submit_button("Submit Data")
            if submit:
                tanaman.insert_tanaman(
                    db=st.session_state["db"],
                    jenis_tanaman=jenis_tanaman,
                    suhu=suhu,
                    suhu_interpolasi=suhu_interpolasi,
                    curah_hujan=curah_hujan,
                    curah_hujan_interpolasi=curah_hujan_interpolasi,
                    kelembapan=kelembapan,
                    kelembapan_interpolasi=kelembapan_interpolasi,
                    jenis_tanah=", ".join(jenis_tanah),
                    tekstur_tanah=", ".join(tekstur_tanah),
                    ph=ph,
                    ph_interpolasi=ph_interpolasi,
                    kemiringan=kemiringan,
                    kemiringan_interpolasi=kemiringan_interpolasi,
                    topografi=topografi,
                    topografi_interpolasi=topografi_interpolasi,
                )

    # READ
    with st.expander("Data Tanaman", expanded=True):
        tanamans = tanaman.read_tanaman(db=st.session_state["db"])
        st.dataframe(tanamans)
    
    # READ
    with st.expander("Update Data Tanaman", expanded=False):
        with st.form("form_update", border=False, clear_on_submit=True):
            st.write("**ID Tanaman**")
            id_tanaman = st.number_input("ID Tanaman", min_value=0, value=None)


            submit_update = st.form_submit_button("Update")
            if submit_update:
                tanaman.delete_tanaman(db=st.session_state["db"], id=id_tanaman)
    
    # DELETEs
    with st.expander("Hapus Tanaman", expanded=False):
        with st.form("form_delete", border=False, clear_on_submit=True):
            st.write("**ID Tanaman**")
            id_tanaman = st.number_input("ID Tanaman", min_value=0, value=None)

            submit_delete = st.form_submit_button("Hapus")
            if submit_delete:
                tanaman.delete_tanaman(db=st.session_state["db"], id=id_tanaman)
        


with tab2:
    # TODO: CRUD profil kondisi lahan
    # CREATE
    with st.expander("Input Data Lahan Baru", expanded=False):
        with st.form("lahan_form", border=False, clear_on_submit=True):
            # create input fields
            lahan_kota = st.text_input("Kota", placeholder="ex: Bawang Merah")
            lahan_suhu = st.text_input("Suhu", placeholder="ex: 10")
            lahan_curah_hujan = st.text_input("Curah Hujan", placeholder="ex: 350")
            lahan_kelembapan = st.text_input("Kelembapan", placeholder="ex: 80")
            lahan_jenis_tanah = st.selectbox("Jenis Tanah", ("mediteran", "regosol", "latosol", "grumosol", "aluvial"))
            lahan_tekstur_tanah = st.selectbox("Tekstur Tanah", ("liat", "liat berpasir", "lempung berpasir sangat halus", "lempung berdebu"))
            lahan_ph = st.text_input("pH", placeholder="ex: 6")
            lahan_kemiringan = st.text_input("Kemiringan", placeholder="ex: 5.5")
            lahan_topografi = st.text_input("Topografi", placeholder="ex: 700")

            # submit btn
            lahan_submit = st.form_submit_button("Submit Data")
            if lahan_submit:
                pass

    # READ
    with st.expander("Data Lahan", expanded=True):
        tanamans = lahan.read_lahan(db=st.session_state["db"])
        st.dataframe(tanamans)
    
    # UPDATE
    with st.expander("Update Data Lahan", expanded=False):
        with st.form("lahan_form_update", border=False, clear_on_submit=True):
            st.write("**ID Lahan**")
            id_tanaman = st.number_input("ID Lahan", min_value=0, value=None)


            submit_update = st.form_submit_button("Update")
            if submit_update:
                tanaman.delete_tanaman(db=st.session_state["db"], id=id_tanaman)
    
    # DELETEs
    with st.expander("Hapus Lahan", expanded=False):
        with st.form("lahan_form_delete", border=False, clear_on_submit=True):
            st.write("**ID Lahan**")
            id_lahan = st.number_input("ID Lahan", min_value=0, value=None)

            submit_delete = st.form_submit_button("Hapus")
            if submit_delete:
                lahan.delete_lahan(db=st.session_state["db"], id=id_lahan)

    