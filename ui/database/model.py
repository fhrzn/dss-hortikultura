from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime

Base = declarative_base()


class Tanaman(Base):
    __tablename__ = "tanaman"

    id = Column(Integer, index=True, primary_key=True, autoincrement=True)
    nama_tanaman = Column(String)
    suhu_ideal = Column(String)
    suhu_batas_interpolasi = Column(String)
    curah_hujan_ideal = Column(String)
    curah_hujan_batas_interpolasi = Column(String)
    kelembaban_ideal = Column(String)
    kelembaban_batas_interpolasi = Column(String)
    ph_ideal = Column(String)
    ph_batas_interpolasi = Column(String)
    kemiringan_ideal = Column(String)
    kemiringan_batas_interpolasi = Column(String)
    topografi_ideal = Column(String)
    topografi_batas_interpolasi = Column(String)
