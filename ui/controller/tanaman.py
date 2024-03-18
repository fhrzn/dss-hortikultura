from utils import crud_tanaman as ctanaman
from database.model import Tanaman
from sqlalchemy.orm import Session
import pandas as pd
import logging 


logger = logging.getLogger(__name__)


def insert_tanaman(
    db: Session,
    nama_tanaman: str,
    suhu_ideal: str,
    suhu_batas_interpolasi: str,
    curah_hujan_ideal: str,
    curah_hujan_batas_interpolasi: str,
    kelembaban_ideal: str,
    kelembaban_batas_interpolasi: str,
    ph_ideal: str,
    ph_batas_interpolasi: str,
    kemiringan_ideal: str,
    kemiringan_batas_interpolasi: str,
    topografi_ideal: str,
    topografi_batas_interpolasi: str,
):
    tanaman = Tanaman(
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

    ctanaman.insert_tanaman(db, tanaman)


def read_tanaman(db: Session):
    tanamans =  ctanaman.read_tanaman(db)

    if tanamans:
        # logger.info(tanamans)
        df_tanaman = pd.DataFrame.from_dict(tanamans)
        
        return df_tanaman

    return None