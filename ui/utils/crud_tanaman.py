from sqlalchemy.orm import Session
from database.model import Tanaman
import logging
from utils import dbutils

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
