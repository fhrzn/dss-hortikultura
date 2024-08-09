import streamlit as st
import logging
from utils import dbutils, dssutils
from controller import lahan, tanaman
import json
import pandas as pd
import os
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt 
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import numpy as np
from matplotlib.ticker import PercentFormatter


logger = logging.getLogger(__name__)

dbutils.init_db()


def calculate_metrics(data, ntop: int = None):
    def _calculate_top_metrics(data, ntop):
        try:
            if len(data['actual']) < ntop:
                ntop = len(data['actual'])
            top_pred = data['predicted'][:ntop]
            top_actual = data['actual'][:ntop]
            intersect = set(top_pred).intersection(set(top_actual))
            return 1 if intersect else 0
        except:
            return 0
    
    top1 = _calculate_top_metrics(data, 1)
    top3 = _calculate_top_metrics(data, 3)
    top4 = _calculate_top_metrics(data, 4)
    top5 = _calculate_top_metrics(data, 5)

    result = {
        'index': ['Top-1', 'Top-3', 'Top-4', 'Top-5'],
        'data': {
            'Accuracy': [top1, top3, top4, top5],
            'Recall': [top1, top3, top4, top5],
            'Precision': [top1, top3, top4, top5],
        }
    }

    return result

def make_matrix(data, ntop):
    top_pred = data['predicted'][:ntop]
    top_actual = data['actual'][:ntop]
    matrix = np.zeros((ntop, ntop))

    intersect = set(data['actual'][:ntop]).intersection(set(data['predicted'][:ntop]))
    for i in intersect:
        i_pred = top_pred.index(i)
        j_pred = top_actual.index(i)
        matrix[i_pred, j_pred] = 1

    df_matrix = pd.DataFrame(matrix, index=top_pred, columns=top_actual)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.heatmap(df_matrix, annot=True, cmap='Blues', cbar=False, ax=ax)  # fmt='.2f' for 2 decimal places
    ax.yaxis.set_ticklabels(top_pred)
    ax.xaxis.set_ticklabels(top_actual)
    ax.set_xlabel("\nTrue Label\n")
    ax.set_ylabel("\nPredicted Label")
    ax.set_title(f"\nTop-{ntop}\n")

    return fig, ax

    


st.set_page_config(
    page_title="Hortikultura Evaluasi per Lahan",
    page_icon="🌱",
    initial_sidebar_state="collapsed",
)

st.write("# Evaluasi di tiap Lahan")


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
        t["lahan"] = kota["lahan"]

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
    cols = ["lahan", "jenis_tanaman", "pm_score"]
    df_rank = df_rank[cols].sort_values("pm_score", ascending=False, ignore_index=True)
    df_rank['rank'] = [i for i in range(1, len(df_rank)+1)]
    # df_rank = df_rank[df_rank['rank'] <= 3]
    df_rank = df_rank[['lahan', 'jenis_tanaman', 'rank']]

    rank_kota.append(df_rank)

df_all = pd.concat(rank_kota, axis=0, ignore_index=True)
df_all = df_all.pivot(index="lahan", columns="rank", values="jenis_tanaman").add_prefix("Rank ").reset_index().rename_axis(None, axis=1)
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
eval_result = []
sample_labels = sorted(pair_grt_pred['Bangkoor']['predicted'])
l2i = {v: k for k, v in enumerate(sample_labels)}
i2l = {k: v for k, v in enumerate(sample_labels)}
cities = ["Semua Kota"] + list(pair_grt_pred.keys())

eval_city = st.selectbox("Pilih Kota", cities, index=None)

if eval_city:
    if eval_city.lower() == "semua kota":
        # metrics = {'accuracy': 0, 'recall': 0, 'precision': 0}
        metrics = {
            'accuracy': {
                'top1': 0,
                'top3': 0,
                'top4': 0,
                'top5': 0,
            },
            'recall': {
                'top1': 0,
                'top3': 0,
                'top4': 0,
                'top5': 0,
            },
            'precision': {
                'top1': 0,
                'top3': 0,
                'top4': 0,
                'top5': 0,
            }
        }
        dfs = []
        for city in cities[1:]:
            _metrics = calculate_metrics(pair_grt_pred[city])
            # dfs.append({"city": city, "df": pd.DataFrame(data=_metrics['data'], index=_metrics['index'])})
            dfs.append({
                "lahan": city,
                "Acc_Top1": _metrics['data']['Accuracy'][0],
                "Acc_Top3": _metrics['data']['Accuracy'][1],
                "Acc_Top4": _metrics['data']['Accuracy'][2],
                "Acc_Top5": _metrics['data']['Accuracy'][3],
                "Prec_Top1": _metrics['data']['Precision'][0],
                "Prec_Top3": _metrics['data']['Precision'][1],
                "Prec_Top4": _metrics['data']['Precision'][2],
                "Prec_Top5": _metrics['data']['Precision'][3],
                "Rec_Top1": _metrics['data']['Recall'][0],
                "Rec_Top3": _metrics['data']['Recall'][1],
                "Rec_Top4": _metrics['data']['Recall'][2],
                "Rec_Top5": _metrics['data']['Recall'][3],
            })
            
            metrics['accuracy']['top1'] += _metrics['data']['Accuracy'][0]
            metrics['accuracy']['top3'] += _metrics['data']['Accuracy'][1]
            metrics['accuracy']['top4'] += _metrics['data']['Accuracy'][2]
            metrics['accuracy']['top5'] += _metrics['data']['Accuracy'][3]

            metrics['recall']['top1'] += _metrics['data']['Recall'][0]
            metrics['recall']['top3'] += _metrics['data']['Recall'][1]
            metrics['recall']['top4'] += _metrics['data']['Recall'][2]
            metrics['recall']['top5'] += _metrics['data']['Recall'][3]

            metrics['precision']['top1'] += _metrics['data']['Precision'][0]
            metrics['precision']['top3'] += _metrics['data']['Precision'][1]
            metrics['precision']['top4'] += _metrics['data']['Precision'][2]
            metrics['precision']['top5'] += _metrics['data']['Precision'][3]

        metrics['accuracy']['top1'] /= len(cities[1:])
        metrics['accuracy']['top3'] /= len(cities[1:])
        metrics['accuracy']['top4'] /= len(cities[1:])
        metrics['accuracy']['top5'] /= len(cities[1:])

        metrics['recall']['top1'] /= len(cities[1:])
        metrics['recall']['top3'] /= len(cities[1:])
        metrics['recall']['top4'] /= len(cities[1:])
        metrics['recall']['top5'] /= len(cities[1:])

        metrics['precision']['top1'] /= len(cities[1:])
        metrics['precision']['top3'] /= len(cities[1:])
        metrics['precision']['top4'] /= len(cities[1:])
        metrics['precision']['top5'] /= len(cities[1:])

        final_data = {
            'Accuracy': [
                metrics['accuracy']['top1'],
                metrics['accuracy']['top3'],
                metrics['accuracy']['top4'],
                metrics['accuracy']['top5'],
            ],
            'Recall': [
                metrics['recall']['top1'],
                metrics['recall']['top3'],
                metrics['recall']['top4'],
                metrics['recall']['top5'],
            ],
            'Precision': [
                metrics['precision']['top1'],
                metrics['precision']['top3'],
                metrics['precision']['top4'],
                metrics['precision']['top5'],
            ],
        }

        st.write('### Rata-rata Top-N Accuracy, Precision, Recall')
        eval_df = pd.DataFrame(data=final_data, index=_metrics['index'])
        
        st.dataframe(eval_df[['Accuracy', 'Recall', 'Precision']].applymap(lambda x: f'{x*100:.1f}%'), use_container_width=True)

        fig = plt.figure(figsize=(10, 6))
        eval_melted = eval_df.reset_index().melt(id_vars="index", var_name="metric", value_name="percentage")
        ax = sns.barplot(data=eval_melted, x="index", y="percentage", hue="metric")
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
        plt.title("\nHasil Rekomendasi Sistem dan Data Dinas Pertanian Kab. Sikka\n")

        top_n_ticks = eval_melted["index"].unique()
        for tick in top_n_ticks:
            subset = eval_melted[eval_melted["index"] == tick]
            max_percentage = subset["percentage"].max()
            ax.text(
                x=subset["index"].iloc[0], 
                y=max_percentage + 0.0035,  # Adjust y position for clarity
                s='{:.1f}%'.format(max_percentage * 100),
                ha="center"
            )
            
        st.pyplot(fig)

        st.write('#### Detil Top-N Accuracy, Precision, Recall per Lahan')
        st.dataframe(pd.DataFrame(data=dfs), use_container_width=True)
        
    else:
        data = pair_grt_pred[eval_city]
        n_top = st.number_input("Top N (default: 3)", min_value=1, value=3, max_value=len(data['actual']))

        st.write('### Top-N Accuracy, Precision, Recall')
        metrics = calculate_metrics(data, n_top)
        st.dataframe(pd.DataFrame(data=metrics['data'], index=metrics['index']), use_container_width=True)
        
        st.write('### Confusion Matrix')
        fig, ax = make_matrix(data, n_top)
        st.pyplot(fig)

        st.write("### Hasil Prediksi vs Ground Truth")
        data['actual'] = data['actual'] + [None] * (len(data['predicted']) - len(data['actual']))
        st.dataframe(pd.DataFrame(data), use_container_width=True, height=600)


