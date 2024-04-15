import streamlit as st
import logging
from utils import dbutils, dssutils
from dotenv import load_dotenv
from controller import lahan, tanaman
import json
import pandas as pd
import os

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


st.write("# 🌱 Hortikultura DSS")

# load data
with open(os.getenv("EIGENVECTOR_JSON_PATH"), "r") as f:
    eigenvectors = json.loads(f.read())

lahans = lahan.read_lahan(db=st.session_state["db"], return_dict=False)
tanamans = tanaman.read_tanaman(db=st.session_state["db"], return_dict=False)

selected_kota = None

st.subheader("Step 1 - Pilih Kota")
with st.expander("Pilih Kota/Desa tempat Lahan", expanded=True):
    kotas = [l['desa'] for l in lahans]
    kota_lahan = st.selectbox("Daftar Kota/Desa", kotas, index=None)

    if kota_lahan:
        selected_kota = [l for l in lahans if l['desa'] == kota_lahan][0]
        # TODO: add catch when selected kota is empty
        # st.write(selected_kota)

        lahan_kota = st.text_input("Kota", disabled=True, value=selected_kota["desa"])
        lahan_suhu = st.text_input("Suhu", disabled=True, value=selected_kota["suhu"])
        lahan_curah_hujan = st.text_input("Curah Hujan", disabled=True, value=selected_kota["curah_hujan"])
        lahan_kelembapan = st.text_input("Kelembapan", disabled=True, value=selected_kota["kelembapan"])
        lahan_jenis_tanah = st.text_input("Jenis Tanah", disabled=True, value=selected_kota["jenis_tanah"])
        lahan_tekstur_tanah = st.text_input("Tekstur Tanah", disabled=True, value=selected_kota["tekstur_tanah"])
        lahan_ph = st.text_input("pH", disabled=True, value=selected_kota["ph"])
        lahan_kemiringan = st.text_input("Kemiringan", disabled=True, value=selected_kota["kemiringan"])
        lahan_topografi = st.text_input("Topografi", disabled=True, value=selected_kota["topografi"])

        # add eigenvector attribute
        selected_kota["ev_jenis_tanah"] = eigenvectors["eigenvector"]["ahp_jenis_tanah"][lahan_jenis_tanah]
        selected_kota["ev_tekstur_tanah"] = eigenvectors["eigenvector"]["ahp_tekstur_tanah"][lahan_tekstur_tanah]

if selected_kota:
    st.subheader("Step 2 - Perhitungan AHP & Interpolasi")
    with st.expander("AHP & Interpolasi", expanded=True):

        st.write("Hasil AHP & Interpolasi")

        # add eigenvector attribute
        for t in tanamans:
            t["ev_jenis_tanah"] = eigenvectors["eigenvector"]["ahp_jenis_tanah"][t["jenis_tanah"].lower()]
            t["ev_tekstur_tanah"] = eigenvectors["eigenvector"]["ahp_tekstur_tanah"][t["tekstur_tanah"].lower()]

            # compute gap value
            t["gap_jenis_tanah"] = selected_kota["ev_jenis_tanah"] - t["ev_jenis_tanah"]
            t["gap_tekstur_tanah"] = selected_kota["ev_tekstur_tanah"] - t["ev_tekstur_tanah"]

            # compute bobot value
            t["bobot_jenis_tanah"] = dssutils.interpolasi_gap(t["gap_jenis_tanah"])
            t["bobot_tekstur_tanah"] = dssutils.interpolasi_gap(t["gap_tekstur_tanah"])

            ###################
            ### INTERPOLASI ###
            ###################
            # suhu
            suhu_intp_range = list(map(int, t["suhu_interpolasi"].split(",")[1:]))
            t["suhu_intp"] = dssutils.interpolate_4_points(selected_kota["suhu"], *suhu_intp_range)

            # curah hujan
            curah_intp_range = list(map(int, t["curah_hujan_interpolasi"].split(",")[1:]))
            t["curah_intp"] = dssutils.interpolate_4_points(selected_kota["curah_hujan"], *curah_intp_range)
            
            # kelembapan
            kelembapan_intp_range = list(map(int, t["kelembapan_interpolasi"].split(",")[1:]))
            t["kelembapan_intp"] = dssutils.interpolate_4_points(selected_kota["kelembapan"], *kelembapan_intp_range)
            
            # ph
            ph_intp_range = list(map(float, t["ph_interpolasi"].split(",")[1:]))
            t["ph_intp"] = dssutils.interpolate_4_points(selected_kota["ph"], *ph_intp_range)
            
            # kemiringan
            kemiringan_intp_range = list(map(int, t["kemiringan_interpolasi"].split(",")[1:]))
            if len(kemiringan_intp_range) < 3:
                t["kemiringan_intp"] = dssutils.interpolate_3_points(selected_kota["kemiringan"], *kemiringan_intp_range)
            else:
                t["kemiringan_intp"] = dssutils.interpolate_4_points(selected_kota["kemiringan"], *kemiringan_intp_range)
            
            # topografi
            topografi_intp_range = list(map(int, t["topografi_interpolasi"].split(",")[1:]))
            t["topografi_intp"] = dssutils.interpolate_4_points(selected_kota["topografi"], *topografi_intp_range)
            
            # calculate nilai sub-kriteria iklim
            t["suhu_nk"] = t["suhu_intp"] * eigenvectors["eigenvector"]["sub_kriteria_iklim"]["suhu"]
            t["kelembapan_nk"] = t["kelembapan_intp"] * eigenvectors["eigenvector"]["sub_kriteria_iklim"]["kelembapan"]
            t["curah_nk"] = t["curah_intp"] * eigenvectors["eigenvector"]["sub_kriteria_iklim"]["curah_hujan"]
            t["iklim_nk"] = t["suhu_nk"] + t["kelembapan_nk"] + t["curah_nk"]

            # calculate nilai sub-kriteria tanah
            t["ph_nk"] = t["ph_intp"] * eigenvectors["eigenvector"]["sub_kriteria_tanah"]["ph"]
            t["kemiringan_nk"] = t["kemiringan_intp"] * eigenvectors["eigenvector"]["sub_kriteria_tanah"]["kemiringan"]
            t["jenis_tanah_nk"] = t["bobot_jenis_tanah"] * eigenvectors["eigenvector"]["sub_kriteria_tanah"]["jenis_tanah"]
            t["tekstur_tanah_nk"] = t["bobot_tekstur_tanah"] * eigenvectors["eigenvector"]["sub_kriteria_tanah"]["tekstur_tanah"]
            t["tanah_nk"] = t["ph_nk"] + t["kemiringan_nk"] + t["jenis_tanah_nk"] + t["tekstur_tanah_nk"]

            # calculate profile matching
            t["pm_score"] = t["iklim_nk"] * eigenvectors["eigenvector"]["kriteria"]["iklim"] + \
                            t["tanah_nk"] * eigenvectors["eigenvector"]["kriteria"]["tanah"] + \
                            t["topografi_intp"] * eigenvectors["eigenvector"]["kriteria"]["topografi"]

            
        df_calc = pd.DataFrame.from_dict(tanamans)
        cols = ["jenis_tanaman", "ev_jenis_tanah", "ev_tekstur_tanah", "gap_jenis_tanah", "gap_tekstur_tanah",
                "bobot_jenis_tanah", "bobot_tekstur_tanah", "suhu_intp", "curah_intp", "kelembapan_intp",
                "ph_intp", "kemiringan_intp", "topografi_intp"]
        df_calc = df_calc[cols]
        st.dataframe(df_calc)


    st.subheader("Step 3 - Hasil Ranking")
    with st.expander("Ranking", expanded=True):

        df_rank = pd.DataFrame.from_dict(tanamans)
        cols = ["jenis_tanaman", "pm_score"]
        df_rank = df_rank[cols].sort_values("pm_score", ascending=False, ignore_index=True)
        df_rank['rank'] = [i for i in range(1, len(df_rank)+1)]
        st.dataframe(df_rank)

    # st.write(tanamans)
        
