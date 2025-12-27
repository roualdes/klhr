from collections import namedtuple
import itertools as it
from pathlib import Path

FILENAME = "experiments.txt"

def named_products(names, *iterables):
    aproduct = namedtuple("aproduct", names)
    prdcts = it.product(*iterables)
    for prdct in prdcts:
        yield aproduct._make(prdct)

def accuracy_experiment():
    """
    Run accuracy experiments

    Stan model: normal.stan

    Check accuracy for 100-D isotropic Normal distribution for KLHR algorithms versus Metropolis
    """

    M = [10_000_000, ]
    algos = ["klhr", "klhr_sub_sinh","slice"]

    itrs = [M, algos]
    names = ["iteration", "algorithm"]

    with open(FILENAME, "a") as f:
        for p in named_products(names, *itrs):
            command = f"uv run experiment_accuracy.py -v "
            command += f"-M {p.iteration} "
            command += f"{p.algorithm}\n"
            f.write(command)

def ar1_experiment():
    """
    Run ar1 experiments

    Stan model: ar1.stan

    Check accuracy for 100-D AR(1) distribution for KLHR algorithms
    """

    algos = ["klhr", "klhr_sub_sinh", "slice"]
    reps = range(20)

    itrs = [algos, reps]
    names = ["algorithm", "rep"]

    with open(FILENAME, "a") as f:
        for p in named_products(names, *itrs):
            command = "uv run experiment_ar1.py -v "
            command += "-M 200_000 "
            command += f"-r {p.rep} "
            command += f"{p.algorithm}\n"
            f.write(command)


def funnel_experiment():
    """
    Run funnel experiments

    Stan model: funnel.stan

    Check depth, breadth, and accuracy for 10-D funnel distribution for KLHR algorithms
    """

    M = [10_000_000,]
    algos = ["klhr", "klhr_sub_sinh", "slice"]

    itrs = [M, algos]
    names = ["iteration", "algorithm"]

    with open(FILENAME, "a") as f:
        for p in named_products(names, *itrs):
            command = f"uv run experiment_funnel.py -v "
            command += f"-M {p.iteration} "
            command += f"{p.algorithm}\n"
            f.write(command)


def relaxation_time_experiment():
    """
    Run relaxation time experiments

    Stan model: earnings.stan
    """

    algos = ["klhr", "klhr_sub_sinh", "klhr_sinh", "slice"]
    tols = [1e-3, 1e-4, 1e-5,]

    itrs = [algos, tols]
    names = ["algorithm", "tolerance"]

    with open(FILENAME, "a") as f:
        for p in named_products(names, *itrs):
            command = "uv run experiment_relaxationtime.py "
            command += "-M 30_000 -w 15_000 -r 20 "
            command += f"-t {p.tolerance} "
            command += f"{p.algorithm}\n"
            f.write(command)

if __name__ == "__main__":
    p = Path(FILENAME)
    p.unlink(missing_ok=True)
    p.touch()
    accuracy_experiment()
    ar1_experiment()
    funnel_experiment()
    relaxation_time_experiment()
    print(f"wrote {FILENAME}")
