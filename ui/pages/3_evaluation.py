import streamlit as st
import logging
from utils import dbutils, dssutils
from controller import lahan, tanaman
import json
import pandas as pd
import os
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt 
import seaborn as sns


logger = logging.getLogger(__name__)

dbutils.init_db()


st.set_page_config(
    page_title="Hortikultura Evaluasi per Kota",
    page_icon="🌱",
    initial_sidebar_state="collapsed",
)

st.write("# Evaluasi di tiap Kota")


# load data
with open(os.getenv("EIGENVECTOR_JSON_PATH"), "r") as f:
    eigenvectors = json.loads(f.read())

lahans = lahan.read_lahan(db=st.session_state["db"], return_dict=False)
tanamans = tanaman.read_tanaman(db=st.session_state["db"], return_dict=False)

rank_kota = []

for kota in lahans:
    # add eigenvector attribute
    kota["ev_jenis_tanah"] = eigenvectors["eigenvector"]["ahp_jenis_tanah"][kota["jenis_tanah"]]
    kota["ev_tekstur_tanah"] = eigenvectors["eigenvector"]["ahp_tekstur_tanah"][kota["tekstur_tanah"]]
    
    tanaman_kota = tanamans.copy()
    for t in tanaman_kota:
        # add kota name
        t["kota"] = kota["desa"]

        # add eigenvector attribute
        t["ev_jenis_tanah"] = eigenvectors["eigenvector"]["ahp_jenis_tanah"][t["jenis_tanah"].lower()]
        t["ev_tekstur_tanah"] = eigenvectors["eigenvector"]["ahp_tekstur_tanah"][t["tekstur_tanah"].lower()]
        
        # compute gap value
        t["gap_jenis_tanah"] = kota["ev_jenis_tanah"] - t["ev_jenis_tanah"]
        t["gap_tekstur_tanah"] = kota["ev_tekstur_tanah"] - t["ev_tekstur_tanah"]

        # compute bobot value
        t["bobot_jenis_tanah"] = dssutils.interpolasi_gap(t["gap_jenis_tanah"])
        t["bobot_tekstur_tanah"] = dssutils.interpolasi_gap(t["gap_tekstur_tanah"])

        ###################
        ### INTERPOLASI ###
        ###################
        # suhu
        suhu_intp_range = list(map(int, t["suhu_interpolasi"].split(",")[1:]))
        t["suhu_intp"] = dssutils.interpolate_4_points(kota["suhu"], *suhu_intp_range)

        # curah hujan
        curah_intp_range = list(map(int, t["curah_hujan_interpolasi"].split(",")[1:]))
        t["curah_intp"] = dssutils.interpolate_4_points(kota["curah_hujan"], *curah_intp_range)
        
        # kelembapan
        kelembapan_intp_range = list(map(int, t["kelembapan_interpolasi"].split(",")[1:]))
        t["kelembapan_intp"] = dssutils.interpolate_4_points(kota["kelembapan"], *kelembapan_intp_range)
        
        # ph
        ph_intp_range = list(map(float, t["ph_interpolasi"].split(",")[1:]))
        t["ph_intp"] = dssutils.interpolate_4_points(kota["ph"], *ph_intp_range)
        
        # kemiringan
        kemiringan_intp_range = list(map(int, t["kemiringan_interpolasi"].split(",")[1:]))
        if len(kemiringan_intp_range) < 3:
            t["kemiringan_intp"] = dssutils.interpolate_3_points(kota["kemiringan"], *kemiringan_intp_range)
        else:
            t["kemiringan_intp"] = dssutils.interpolate_4_points(kota["kemiringan"], *kemiringan_intp_range)
        
        # topografi
        topografi_intp_range = list(map(int, t["topografi_interpolasi"].split(",")[1:]))
        t["topografi_intp"] = dssutils.interpolate_4_points(kota["topografi"], *topografi_intp_range)
        
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

    # ranking    
    df_rank = pd.DataFrame.from_dict(tanaman_kota)
    cols = ["kota", "jenis_tanaman", "pm_score"]
    df_rank = df_rank[cols].sort_values("pm_score", ascending=False, ignore_index=True)
    df_rank['rank'] = [i for i in range(1, len(df_rank)+1)]
    # df_rank = df_rank[df_rank['rank'] <= 3]
    df_rank = df_rank[['kota', 'jenis_tanaman', 'rank']]

    rank_kota.append(df_rank)

df_all = pd.concat(rank_kota, axis=0, ignore_index=True)
df_all = df_all.pivot(index="kota", columns="rank", values="jenis_tanaman").add_prefix("Rank ").reset_index().rename_axis(None, axis=1)
# df_all['predicted'] = df_all['Rank 1']
# df_all = pd.pivot_table(df_all, index="kota", columns="jenis_tanaman")
# st.dataframe(df_all, use_container_width=True, height=1000)

# display evaluation on each city
df_grt = pd.read_csv("./data/ground_truth.csv", sep=';')

# data pairing
pair_grt_pred = {}
for _, data in df_all.iterrows():
    d = data.to_list()
    pair_grt_pred[d[0]] = {}
    pair_grt_pred[d[0]]["predicted"] = [i.lower() for i in d[1:]]
    pair_grt_pred[d[0]]["actual"] = df_grt[df_grt["kota"] == d[0]]["tanaman"].item().split(", ")
    # st.write(d[0])

# evaluation
l2i = {}
i2l = {}
cities = list(pair_grt_pred.keys())

eval_city = st.selectbox("Pilih Kota", cities, index=None)

if eval_city:
    data = pair_grt_pred[eval_city]
    if not l2i:
        labels = sorted(data["predicted"])
        l2i = {v: k for k, v in enumerate(labels)}
        i2l = {k: v for k, v in enumerate(labels)}

    y_true = [l2i[i] for i in data["actual"]][:3]
    y_pred = [l2i[i] for i in data["predicted"] if l2i[i] in y_true]
    y_label = [i2l[i] for i in y_true]

    st.write(f"Accuracy Score: {accuracy_score(y_true, y_pred) * 100:.2f}%")
    st.write(f"Recall Score: {recall_score(y_true, y_pred, average='weighted') * 100:.2f}%")
    st.write(f"Precision Score: {precision_score(y_true, y_pred, average='weighted') * 100:.2f}%")
    

    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    ax = sns.heatmap(cm, annot=True, fmt='d')
    ax.xaxis.set_ticklabels(y_label)
    ax.yaxis.set_ticklabels(y_label)
    ax.set_xlabel("\nPredicted Label")
    ax.set_ylabel("True Label\n")
    ax.set_title("\nConfusion Matrix\n")

    st.pyplot(fig)


    

