import duckdb
import numpy as np
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

def dquote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'

def write_table(table: str, cols: list[tuple[str, str]]) -> str:
    ddl = f"CREATE TABLE IF NOT EXISTS {dquote(table)} (\n"
    for col in cols:
        c, t = col
        ddl += f"{dquote(c)} {t},\n"
    return ddl + ");"

def init_table(db_path: str,
               table: str,
               cols: list[tuple[str, str]],
               start_fresh: bool = False):
    con = connect_db(db_path)
    if start_fresh:
        con.sql(f"DROP TABLE IF EXISTS {dquote(table)};")
    con.sql(write_table(table, cols))
    con.close()

def dot_to_underscore(s: str) -> str:
    return s.replace(".", "_")

def make_table(dbpath, bsm, tbl, start_fresh):
    tbl_cols = [
        ("algorithm",           "VARCHAR"),
        ("model",               "VARCHAR"),
        ("replication",         "UINT32"),
        ("acceptance_rate",     "DOUBLE"),
        ("msjd",                "DOUBLE"),
        ("ld_evals",            "DOUBLE"),
        ("runtime",             "DOUBLE"),
        ("num_params",          "UINT32"),
    ]
    varnames = list(map(dot_to_underscore, bsm.parameter_names()))
    for var in varnames:
        tbl_cols += [
            (var + "m",         "DOUBLE"),
            (var + "m2",        "DOUBLE"),
            (var + "s",         "DOUBLE"),
            (var + "s2",        "DOUBLE"),
        ]
    init_table(dbpath, tbl, tbl_cols, start_fresh)


def update_table(dbpath, tbl, algorithm, algo, draws, warmup, rep, runtime):
    M = np.shape(draws)[0]
    msjd = 0.0
    for m in range(M-1):
        d = np.linalg.norm(draws[m+1] - draws[m]) - msjd
        msjd += d / (m + 1)
    ldevals = algo.ld_evals if algorithm == "slice" else algo.grad_evals
    d = {
        "algorithm": algorithm,
        "replication": rep,
        "msjd": msjd,
        "acceptance_rate": algo.acceptance_probability,
        "ld_evals": ldevals,
        "runtime": runtime,
    }

    m = np.mean(draws[warmup:, :], axis = 0)
    v = np.std(draws[warmup:, :], ddof = 1, axis = 0)
    varnames = list(map(dot_to_underscore, algo.model.parameter_names()))
    d |= {varnames[i] + "m": mi for i, mi in enumerate(m)}
    d |= {varnames[i] + "s": vi for i, vi in enumerate(v)}
    append_df(dbpath, tbl, d)
