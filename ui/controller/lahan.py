from utils import crud_lahan as clahan
from database.model import Lahan
from sqlalchemy.orm import Session
import pandas as pd
import logging 


logger = logging.getLogger(__name__)


def insert_lahan(
    db: Session,
    desa: str,
    suhu: str,
    curah_hujan: str,
    kelembapan: str,
    jenis_tanah: str,
    tekstur_tanah: str,
    ph: str,
    kemiringan: str,
    topografi: str,
):
    lahan = Lahan(
        desa=desa,
        suhu=suhu,
        curah_hujan=curah_hujan,
        kelembapan=kelembapan,
        jenis_tanah=jenis_tanah,
        tekstur_tanah=tekstur_tanah,
        ph=ph,
        kemiringan=kemiringan,
        topografi=topografi,
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


def delete_lahan(db: Session, id: int):
    clahan.delete_lahan(db, id)