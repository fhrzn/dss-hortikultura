from sqlalchemy.orm import Session
from database.model import Lahan
import logging
from utils import dbutils
import string

logger = logging.getLogger(__name__)

def insert_lahan(db: Session, lahan: Lahan):
    try:
        db.add(lahan)
        db.commit()
        db.refresh(lahan)
    except Exception as e:
        logger.error((str(e)))
        logger.error("Rollback database caused by prior error message.")
        db.rollback()


def read_lahan(db: Session):
    try:
        query = db.query(Lahan)
        return dbutils.sqlalchemy_to_dict(query.all())
    except Exception as e:
        logger.error(str(e))



def read_lahan_by_name(db: Session, name: str):
    try:
        names_ = (name, name.lower(), name.upper(), name.capitalize(), name.title(), string.capwords(name))
        logger.info(f"Checking {names_} in database")
        query = db.query(Lahan).filter(Lahan.desa.in_(names_))
        return dbutils.sqlalchemy_to_dict(query.first())
    except Exception as e:
        logger.error(str(e))






def delete_lahan(db: Session, id: int):
    try:
        lahan = db.query(Lahan).where(Lahan.id == id).first()
        
        db.delete(lahan)
        db.commit()
    except Exception as e:
        logger.error(str(e))