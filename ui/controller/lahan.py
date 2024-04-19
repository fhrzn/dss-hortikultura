from utils import crud_lahan as clahan
from database.model import Lahan
from sqlalchemy.orm import Session
import pandas as pd
import logging 


logger = logging.getLogger(__name__)


def insert_lahan(
    db: Session,
    desa: str,
    curah_hujan: str,
    jenis_tanah: str,
    tekstur_tanah: str,
    suhu: float,
    kelembapan: float,
    ph: float,
    kemiringan: float,
    topografi: float,
):
    lahan = Lahan(
        desa=desa,
        suhu=float(suhu),
        curah_hujan=float(curah_hujan),
        kelembapan=float(kelembapan),
        jenis_tanah=jenis_tanah,
        tekstur_tanah=tekstur_tanah,
        ph=float(ph),
        kemiringan=float(kemiringan),
        topografi=float(topografi),
    )

    clahan.insert_lahan(db, lahan)


def read_lahan(db: Session, return_dict: bool = True):
    lahans =  clahan.read_lahan(db)

    if lahans:
        if return_dict:
            df_lahan = pd.DataFrame.from_dict(lahans)
            return df_lahan
        
        return lahans
        
    return None


def get_lahan_by_name(db: Session, name: str):
    lahan = clahan.read_lahan_by_name(db, name)
    return lahan


def delete_lahan(db: Session, id: int):
    clahan.delete_lahan(db, id)