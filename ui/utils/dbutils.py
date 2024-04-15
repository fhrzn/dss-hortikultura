from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
import os
from database import model
import streamlit as st
import logging
from dotenv import load_dotenv


load_dotenv()
logger = logging.getLogger(__name__)


def init_db():
    # TODO: add replace arguments to force re-create db
    # check session variable
    if "db" not in st.session_state:
        engine = create_engine(os.getenv("SQLITE_URL"))
        session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = session()

        # check db availability
        if not os.path.exists(os.getenv("DB_PATH")):
            model.Base.metadata.create_all(engine)
            execute_seeds(db)
        
        # assign to session variable
        st.session_state["db"] = db


def execute_seeds(db: Session):
    for seed in os.listdir(os.getenv("SEED_DIR")):
        if '.sql' in seed:
            with open(os.path.join(os.getenv("SEED_DIR"), seed), 'r', encoding='utf-8') as f:
                try:
                    logger.info(f"Found {seed}")
                    queries = f.readlines()
                    for q in queries:
                        db.execute(text(q))
                        db.commit()
                    logger.info("Seeds %s has executed successfully." % seed)
                except Exception as e:
                    db.rollback()
                    logger.error(str(e))



def _sqlalchemy_query_to_dict(item):

    d = {}
    for col in item.__table__.columns:
        d[col.name] = getattr(item, col.name)

    return d


def sqlalchemy_to_dict(item):
    if isinstance(item, list):
        return [_sqlalchemy_query_to_dict(res) for res in item]
    else:
        return _sqlalchemy_query_to_dict(item)