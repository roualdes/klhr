DESCRIBE relaxationtime;

SELECT algorithm,
       count(*),
       avg(mb0)::DECIMAL(8, 2) as mb0,
       avg(mb1)::DECIMAL(6, 2) as mb1,
       -- avg(msigma)::DECIMAL(3, 2) as msigma,
       -- avg(ms)::DECIMAL(3, 2) as ms,
       avg(acceptance_rate)::DECIMAL(3, 2) as a,
       avg(msjd)::DECIMAL(5, 2) as msjd,
       avg(runtime)::DECIMAL(5, 2) as r,
       avg(ld_evals)::DECIMAL(10, 0) as ld,
FROM relaxationtime
WHERE mb0 < -5e4 AND -6e4 < mb0
GROUP BY algorithm
ORDER BY algorithm;
