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
    algos = ["klhr", "klhr_sinh"]
    warmups = [0, M[0] // 2]
    scale_dir_covs = ["-s", " "]
    overrelaxed_flags = ["-o", " "]
    eigen_method_one_flags = ["-e1", " "]

    itrs = [
        M,
        warmups,
        scale_dir_covs,
        overrelaxed_flags,
        eigen_method_one_flags,
        algos
    ]

    names = [
        "iteration",
        "warmup",
        "scale_cov",
        "overrelaxed",
        "eigen",
        "algorithm"
    ]

    with open(FILENAME, "a") as f:
        for p in named_products(names, *itrs):
            command = f"python experiment_accuracy.py "
            command += f"-M {p.iteration} "
            command += f"-w {p.warmup} "
            command += f"{p.scale_cov} "
            command += f"{p.overrelaxed} "
            command += f"{p.eigen} "
            command += f"{p.algorithm}\n"
            f.write(command)

def ar1_experiment():
    """
    Run ar1 experiments

    Stan model: ar1.stan

    Check accuracy for 100-D AR(1) distribution for KLHR algorithms

    windowsize -- the size of the initial window in windowed adaptation
    windowscale -- the amount by which to increase the window size after each closed window
    J -- the number of eigenvector to estimate
    """

    windowsizes = [25, 50]
    windowscales = [2,]
    Js = [2, 4, 8, 10]
    Ls = [0, 2, 4]
    scale_dir_covs = ["-s", " "]
    overrelaxed_flags = ["-o", " "]
    eigen_method_one_flags = ["-e1", " "]
    algos = ["klhr", "klhr_sinh"]
    reps = range(10)

    itrs = [
        windowsizes,
        windowscales,
        Js,
        Ls,
        scale_dir_covs,
        overrelaxed_flags,
        eigen_method_one_flags,
        algos,
        reps
    ]

    names = [
        "windowsize",
        "windowscale",
        "J",
        "L",
        "scale_cov",
        "overrelaxed",
        "eigen",
        "algorithm",
        "rep"
    ]

    with open(FILENAME, "a") as f:
        for p in named_products(names, *itrs):
            command = "python experiment_ar1.py "
            command += "-M 100_000 -w 50_000 "
            command += f" --windowsize {p.windowsize} "
            command += f"--windowscale {p.windowscale} "
            command += f"-J {p.J} "
            command += f"-l {p.L} "
            command += f"-r {p.rep} "
            command += f"{p.scale_cov} "
            command += f"{p.overrelaxed} "
            command += f"{p.eigen} "
            command += f"{p.algorithm}\n"
            f.write(command)


def funnel_experiment():
    """
    Run funnel experiments

    Stan model: funnel.stan

    Check depth, breadth, and accuracy for 10-D funnel distribution for KLHR algorithms
    """

    M = [10_000_000,]
    warmups = [0, M[0] // 2]
    scale_dir_covs = ["-s", " "]
    overrelaxed_flags = ["-o", " "]
    eigen_method_one_flags = ["-e1", " "]
    algos = ["klhr", "klhr_sinh"]

    itrs = [
        M,
        warmups,
        scale_dir_covs,
        overrelaxed_flags,
        eigen_method_one_flags,
        algos,
    ]

    names = [
        "iteration",
        "warmup",
        "scale_cov",
        "overrelaxed",
        "eigen",
        "algorithm",
    ]

    with open(FILENAME, "a") as f:
        for p in named_products(names, *itrs):
            command = f"python experiment_funnel.py "
            command += f"-M {p.iteration} "
            command += f"-w {p.warmup} "
            command += f"{p.scale_cov} "
            command += f"{p.overrelaxed} "
            command += f"{p.eigen} "
            command += f"{p.algorithm}\n"
            f.write(command)


def relaxation_time_experiment():
    """
    Run relaxation time experiments

    Stan model: earnings.stan

    Explore the effect on relaxation time of the following parameters:

    windowsize -- the size of the initial window in windowed adaptation
    windowscale -- the amount by which to increase the window size after each closed window
    J -- the number of eigenvector to estimate

    """

    windowsizes = [25, 50]
    windowscales = [2,]
    Js = [2, 3]
    Ls = [0, 2, 4]
    scale_dir_covs = ["-s", " "]
    overrelaxed_flags = ["-o", " "]
    eigen_method_one_flags = ["-e1", " "]
    algos = ["klhr", "klhr_sinh"]
    reps = range(10)

    itrs = [
        windowsizes,
        windowscales,
        Js,
        Ls,
        scale_dir_covs,
        overrelaxed_flags,
        eigen_method_one_flags,
        algos,
        reps,
    ]

    names = [
        "windowsize",
        "windowscale",
        "J",
        "L",
        "scale_cov",
        "overrelaxed",
        "eigen",
        "algorithm",
        "rep",
    ]

    with open(FILENAME, "a") as f:
        for p in named_products(names, *itrs):
            command = "python experiment_relaxationtime.py "
            command += "-M 30_000 -w 15_000 "
            command += f"--windowsize {p.windowsize} "
            command += f"--windowscale {p.windowscale} "
            command += f"-J {p.J} "
            command += f"-l {p.L} "
            command += f"-r {p.rep} "
            command += f"{p.scale_cov} "
            command += f"{p.overrelaxed} "
            command += f"{p.eigen} "
            command += f"{p.algorithm}\n"
            f.write(command)


if __name__ == "__main__":
    Path(FILENAME).touch(exist_ok=False)
    accuracy_experiment()
    ar1_experiment()
    funnel_experiment()
    relaxation_time_experiment()
    print(f"wrote {FILENAME}")
