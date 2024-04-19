from sqlalchemy.orm import Session
from database.model import Tanaman
import logging
from utils import dbutils
import string

logger = logging.getLogger(__name__)

def insert_tanaman(db: Session, tanaman: Tanaman):
    try:
        db.add(tanaman)
        db.commit()
        db.refresh(tanaman)
    except Exception as e:
        logger.error((str(e)))
        logger.error("Rollback database caused by prior error message.")
        db.rollback()


def read_tanaman(db: Session):
    try:
        query = db.query(Tanaman)
        return dbutils.sqlalchemy_to_dict(query.all())
    except Exception as e:
        logger.error(str(e))


def read_tanaman_by_name(db: Session, name: str):
    try:
        names_ = (name, name.lower(), name.upper(), name.capitalize(), name.title(), string.capwords(name))
        logger.info(f"Checking {names_} in database")
        query = db.query(Tanaman).filter(Tanaman.jenis_tanaman.in_(names_))
        return dbutils.sqlalchemy_to_dict(query.first())
    except Exception as e:
        logger.error(str(e))


def delete_tanaman(db: Session, id: int):
    try:
        tanaman = db.query(Tanaman).where(Tanaman.id == id).first()
        
        db.delete(tanaman)
        db.commit()
    except Exception as e:
        logger.error(str(e))