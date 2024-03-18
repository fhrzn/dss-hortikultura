from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime

Base = declarative_base()


class Tanaman(Base):
    __tablename__ = "tanaman"

    id = Column(Integer, index=True, primary_key=True, autoincrement=True)
    jenis_tanaman = Column(String)
    suhu = Column(String)
    suhu_interpolasi = Column(String)
    curah_hujan = Column(String)
    curah_hujan_interpolasi = Column(String)
    kelembapan = Column(String)
    kelembapan_interpolasi = Column(String)
    jenis_tanah = Column(String)
    tekstur_tanah = Column(String)
    ph = Column(String)
    ph_interpolasi = Column(String)
    kemiringan = Column(String)
    kemiringan_interpolasi = Column(String)
    topografi = Column(String)
    topografi_interpolasi = Column(String)

