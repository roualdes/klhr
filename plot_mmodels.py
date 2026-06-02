import warnings

import duckdb
import numpy as np
import pandas as pd

def rmse(mhat, m, s):
    return np.sqrt(np.mean(((mhat - m) / s) ** 2))

con = duckdb.connect("experiments.db")
df = con.sql("""SELECT * FROM lotkavolterra""").df()

adx = df.columns.str.contains('algorithm')
mdx = df.columns.str.contains(r'^(?!algorithm$).*m$', regex=True)
sdx = df.columns.str.contains(f'^(?!(ld_evals|min_ess|num_params)).*s$', regex=True)
m2dx = df.columns.str.contains(r'.*m2$', regex=True)

s2dx = df.columns.str.contains(r'.*s2$', regex=True)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    sdf = (df
           .loc[:, (adx | mdx | m2dx | sdx | s2dx)]
           .groupby("algorithm", as_index = False)
           .agg(np.mean))

stan = sdf["algorithm"] == "stan"
klhr = sdf["algorithm"] == "klhr"
klss = sdf["algorithm"] == "klhr_sub_sinh"
slic = sdf["algorithm"] == "slice"

mdx = sdf.columns.str.contains(r'^(?!algorithm$).*m$', regex=True)
sdx = sdf.columns.str.contains(f'^(?!ld_evals).*s$', regex=True)
m2dx = sdf.columns.str.contains(r'.*m2$', regex=True)
s2dx = sdf.columns.str.contains(r'.*s2$', regex=True)

stan_m = sdf.loc[stan, mdx].values.flatten()
stan_s = sdf.loc[stan, sdx].values.flatten()
stan_m2 = sdf.loc[stan, m2dx].values.flatten()
stan_s2 = sdf.loc[stan, s2dx].values.flatten()

klhr_m = sdf.loc[klhr, mdx].values.flatten()
klhr_s = sdf.loc[klhr, sdx].values.flatten()
klhr_m2 = sdf.loc[klhr, m2dx].values.flatten()
klhr_s2 = sdf.loc[klhr, s2dx].values.flatten()

klss_m = sdf.loc[klss, mdx].values.flatten()
klss_s = sdf.loc[klss, sdx].values.flatten()
klss_m2 = sdf.loc[klss, m2dx].values.flatten()
klss_s2 = sdf.loc[klss, s2dx].values.flatten()

slic_m = sdf.loc[slic, mdx].values.flatten()
slic_s = sdf.loc[slic, sdx].values.flatten()
slic_m2 = sdf.loc[slic, m2dx].values.flatten()
slic_s2 = sdf.loc[slic, s2dx].values.flatten()


print(klhr_m)
print(stan_m)
print(stan_s)

klhr_m_rmse = np.round(rmse(klhr_m, stan_m, stan_s), 4)
klss_m_rmse = np.round(rmse(klss_m, stan_m, stan_s), 4)
slic_m_rmse = np.round(rmse(slic_m, stan_m, stan_s), 4)


print(f"klhr_m: {klhr_m_rmse}")
print(f"klss_m: {klss_m_rmse}")
print(f"slic_m: {slic_m_rmse}")

klhr_m2_rmse = np.round(rmse(klhr_m2, stan_m2, stan_s2), 4)
klss_m2_rmse = np.round(rmse(klss_m2, stan_m2, stan_s2), 4)
slic_m2_rmse = np.round(rmse(slic_m2, stan_m2, stan_s2), 4)

print()
print(f"klhr_m2: {klhr_m2_rmse}")
print(f"klss_m2: {klss_m2_rmse}")
print(f"slic_m2: {slic_m2_rmse}")

print()
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    print(df
          .loc[:, ["algorithm", "acceptance_rate", "ld_evals", "runtime"]]
          .groupby("algorithm", as_index = False)
          .agg(np.mean))
