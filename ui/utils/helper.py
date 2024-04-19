import streamlit as st
from controller import tanaman, lahan
import logging

logger = logging.getLogger(__name__)


def check_tanaman_duplicate(jenis_tanaman: str = None):
    st.session_state["tanaman_duplicate"] = False
    tanam = tanaman.get_tanaman_by_name(st.session_state["db"], jenis_tanaman if jenis_tanaman else st.session_state["jenis_tanaman"])
    logger.info(tanam)
    if tanam:
        st.session_state["tanaman_duplicate"] = True
        return True
    
    return False


def check_lahan_duplicate(kota: str = None):
    st.session_state["lahan_duplicate"] = False
    tanam = lahan.get_lahan_by_name(st.session_state["db"], kota if kota else st.session_state["kota"])
    logger.info(tanam)
    if tanam:
        st.session_state["lahan_duplicate"] = True
        return True
    
    return False


def clear_tanaman_form():
    st.session_state["jenis_tanaman"] = ""
    st.session_state["suhu"] = ""
    st.session_state["suhu_interpolasi"] = ""
    st.session_state["curah_hujan"] = ""
    st.session_state["curah_hujan_interpolasi"] = ""
    st.session_state["kelembapan"] = ""
    st.session_state["kelembapan_interpolasi"] = ""
    st.session_state["jenis_tanah"] = ""
    st.session_state["tekstur_tanah"] = ""
    st.session_state["ph"] = ""
    st.session_state["ph_interpolasi"] = ""
    st.session_state["kemiringan"] = ""
    st.session_state["kemiringan_interpolasi"] = ""
    st.session_state["topografi"] = ""
    st.session_state["topografi_interpolasi"] = ""
    
    
def submit_tanaman():
    # check if there is duplicate entry
    is_duplicate = check_tanaman_duplicate()

    if not is_duplicate:
        tanaman.insert_tanaman(
            db=st.session_state["db"],
            jenis_tanaman=st.session_state["jenis_tanaman"],
            suhu=st.session_state["suhu"],
            suhu_interpolasi=st.session_state["suhu_interpolasi"],
            curah_hujan=st.session_state["curah_hujan"],
            curah_hujan_interpolasi=st.session_state["curah_hujan_interpolasi"],
            kelembapan=st.session_state["kelembapan"],
            kelembapan_interpolasi=st.session_state["kelembapan_interpolasi"],
            jenis_tanah=st.session_state["jenis_tanah"],
            tekstur_tanah=st.session_state["tekstur_tanah"],
            ph=st.session_state["ph"],
            ph_interpolasi=st.session_state["ph_interpolasi"],
            kemiringan=st.session_state["kemiringan"],
            kemiringan_interpolasi=st.session_state["kemiringan_interpolasi"],
            topografi=st.session_state["topografi"],
            topografi_interpolasi=st.session_state["topografi_interpolasi"],
        )

        clear_tanaman_form()