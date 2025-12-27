import duckdb
import pandas as pd
import time

def connect_db(
    db_path: str,
    max_attempts: int = 50,
    minsleep: float = 0.05
):
    for attempt in range(max_attempts):
        try:
            return duckdb.connect(db_path)
        except duckdb.IOException as e:
            msg = str(e)
            is_lock_error = (
                "Could not set lock on file" in msg
                or "Conflicting lock is held" in msg
            )
            if not is_lock_error or attempt == max_attempts:
                raise
            time.sleep(minsleep * (attempt + 1))


def append_df(
    db_path: str,
    tbl: str,
    d: dict
):
    con = connect_db(db_path)
    con.append(tbl, pd.DataFrame([d]), by_name = True)
    con.close()

def init_accuracy(db_path: str, start_fresh: bool = False):
    con = connect_db("experiments.db")
    if start_fresh:
        con.sql("DROP TABLE IF EXISTS accuracy;")
    con.sql("""CREATE TABLE IF NOT EXISTS accuracy (
        algorithm           VARCHAR,
        acceptance_rate     DOUBLE,
        msjd                DOUBLE,
        ld_evals            UINT64,
        runtime             DOUBLE,
    );""")
    con.close()

def init_funnel(db_path: str, start_fresh: bool = False):
    con = connect_db("experiments.db")
    if start_fresh:
        con.sql("DROP TABLE IF EXISTS funnel;")
    con.sql("""CREATE TABLE IF NOT EXISTS funnel (
        algorithm           VARCHAR,
        acceptance_rate     DOUBLE,
        msjd                DOUBLE,
        ld_evals            UINT64,
        runtime             DOUBLE,
    );""")
    con.close()

def init_relaxationtime(db_path: str, start_fresh: bool = False):
    con = connect_db("experiments.db")
    if start_fresh:
        con.sql("DROP TABLE IF EXISTS relaxationtime;")
    con.sql("""CREATE TABLE IF NOT EXISTS relaxationtime (
        algorithm           VARCHAR,
        replication         UINT32,
        acceptance_rate     DOUBLE,
        msjd                DOUBLE,
        mb0                 DOUBLE,
        mb1                 DOUBLE,
        msigma              DOUBLE,
        ms                  DOUBLE,
        vb0                 DOUBLE,
        vb1                 DOUBLE,
        vsigma              DOUBLE,
        vs                  DOUBLE,
        ld_evals            UINT64,
        runtime             DOUBLE,
    );""")
    con.close()

def init_ar1(db_path: str, start_fresh: bool = False):
    con = connect_db("experiments.db")
    if start_fresh:
        con.sql("DROP TABLE IF EXISTS ar1;")
    con.sql("""CREATE TABLE IF NOT EXISTS ar1 (
        algorithm       VARCHAR,
        alpha           DOUBLE,
        acceptance_rate DOUBLE,
        msjd            DOUBLE,
        max_dist_mean   DOUBLE,
        max_dist_var    DOUBLE,
        prop_mean_g0    DOUBLE,
        prop_var_g1     DOUBLE,
        m1              DOUBLE,
        v1              DOUBLE,
        ld_evals        UINT64,
        runtime         DOUBLE,
    );""")
    con.close()
