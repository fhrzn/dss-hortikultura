from sqlalchemy.orm import Session
from database.model import Lahan
import logging
from utils import dbutils

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


def delete_lahan(db: Session, id: int):
    try:
        lahan = db.query(Lahan).where(Lahan.id == id).first()
        
        db.delete(lahan)
        db.commit()
    except Exception as e:
        logger.error(str(e))