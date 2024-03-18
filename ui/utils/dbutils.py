from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from database import model
import streamlit as st


def init_db():
    # check session variable
    if "db" not in st.session_state:
        engine = create_engine(os.getenv("SQLITE_URL"))
        session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = session()

        # check db availability
        if not os.path.exists(os.getenv("SQLITE_URL")):
            model.Base.metadata.create_all(engine)
        
        # assign to session variable
        st.session_state["db"] = db

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