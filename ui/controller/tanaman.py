from utils import crud_tanaman as ctanaman
from database.model import Tanaman
from sqlalchemy.orm import Session
import pandas as pd
import logging 


logger = logging.getLogger(__name__)


def insert_tanaman(
    db: Session,
    jenis_tanaman: str,
    suhu: str,
    suhu_interpolasi: str,
    curah_hujan: str,
    curah_hujan_interpolasi: str,
    kelembapan: str,
    kelembapan_interpolasi: str,
    jenis_tanah: str,
    tekstur_tanah: str,
    ph: str,
    ph_interpolasi: str,
    kemiringan: str,
    kemiringan_interpolasi: str,
    topografi: str,
    topografi_interpolasi: str,
):
    tanaman = Tanaman(
        jenis_tanaman=jenis_tanaman,
        suhu=suhu,
        suhu_interpolasi=suhu_interpolasi,
        curah_hujan=curah_hujan,
        curah_hujan_interpolasi=curah_hujan_interpolasi,
        kelembapan=kelembapan,
        kelembapan_interpolasi=kelembapan_interpolasi,
        jenis_tanah=jenis_tanah,
        tekstur_tanah=tekstur_tanah,
        ph=ph,
        ph_interpolasi=ph_interpolasi,
        kemiringan=kemiringan,
        kemiringan_interpolasi=kemiringan_interpolasi,
        topografi=topografi,
        topografi_interpolasi=topografi_interpolasi,
    )

    ctanaman.insert_tanaman(db, tanaman)


def read_tanaman(db: Session, return_dict: bool = True):
    tanamans =  ctanaman.read_tanaman(db)

    if tanamans:
        if return_dict:
            df_tanaman = pd.DataFrame.from_dict(tanamans)
            return df_tanaman

        return tanamans

    return None


def get_tanaman_by_name(db: Session, name: str):
    tanaman = ctanaman.read_tanaman_by_name(db, name)
    return tanaman


def delete_tanaman(db: Session, id: int):
    ctanaman.delete_tanaman(db, id)