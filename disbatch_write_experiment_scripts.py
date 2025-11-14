import click

@click.command()
@click.option("--accuracy",  default = True, is_flag = True)
@click.option("--ar1",  default = True, is_flag = True)
@click.option("--funnel",  default = True, is_flag = True)
@click.option("--relaxation", default = True, is_flag = True)
def main(relaxation, accuracy, funnel, ar1):
    if accuracy: accuracy_experiment()
    if ar1: ar1_experiment()
    if funnel: funnel_experiment()
    if relaxation: relaxation_time_experiment()


def accuracy_experiment():
    """
    Run accuracy experiments

    Stan model: normal.stan

    Check accuracy for 100-D isotropic Normal distribution for KLHR algorithms versus Metropolis
    """
    filename = "experiments_accuracy"

    base = "python experiment_accuracy.py"

    wd = "experiments/accuracy/"

    prefix = f"#DISBATCH PREFIX ( "
    prefix += f"source experiment_setup ; {base}\n"

    suffix = "#DISBATCH SUFFIX ) &> experiments/${DISBATCH_NAMETASKS}_${DISBATCH_JOBID}_${DISBATCH_TASKID_ZP}.log\n"

    M = 10_000_000
    algos = ["klhr", "klhrsinh"]
    warmups = [0, M // 2]
    scale_dir_covs = ["-s", ""]
    overrelaxed_flags = ["-o", ""]
    eigen_method_one_flags = ["-e1", ""]

    with open(filename, "w") as f:
        f.write(prefix)
        f.write(suffix)
        for algo in algos:
            for warmup in warmups:
                for scale_dir_cov in scale_dir_covs:
                    for overrelaxed_flag in overrelaxed_flags:
                        for eigen_method_one_flag in eigen_method_one_flags:
                            command = f" -M {M} -w {warmup} {scale_dir_cov} {overrelaxed_flag} {eigen_method_one_flag} {algo}\n"
                            f.write(command)

    print(f"wrote file: {filename}")


def ar1_experiment():
    """
    Run ar1 experiments

    Stan model: ar1.stan

    Check accuracy for 100-D AR(1) distribution for KLHR algorithms

    windowsize -- the size of the initial window in windowed adaptation
    windowscale -- the amount by which to increase the window size after each closed window
    J -- the number of eigenvector to estimate
    """
    filename = "experiments_ar1"

    base = "python experiment_ar1.py -M 100_000 -w 50_000"

    wd = "experiments/ar1/"

    prefix = f"#DISBATCH PREFIX ( "
    prefix += f"source experiment_setup ; {base}\n"

    suffix = "#DISBATCH SUFFIX ) &> experiments/${DISBATCH_NAMETASKS}_${DISBATCH_JOBID}_${DISBATCH_TASKID_ZP}.log\n"

    windowsizes = [25, 50]
    windowscales = [2]
    Js = [2, 4, 8, 10]
    Ls = [0, 2, 4]
    scale_dir_covs = ["-s", ""]
    overrelaxed_flags = ["-o", ""]
    eigen_method_one_flags = ["-e1", ""]
    algos = ["klhr", "klhrsinh"]
    reps = range(10)

    with open(filename, "w") as f:
        f.write(prefix)
        f.write(suffix)
        for wsize in windowsizes:
            for wscale in windowscales:
                for j in Js:
                    for l in Ls:
                        for algo in algos:
                            for r in reps:
                                for scale_dir_cov in scale_dir_covs:
                                    for overrelaxed_flag in overrelaxed_flags:
                                        for eigen_method_one_flag in eigen_method_one_flags:
                                            command = f" --windowsize {wsize} "
                                            command += f"--windowscale {wscale} "
                                            command += f"-J {j} "
                                            command += f"-l {l} "
                                            command += f"-r {r} "
                                            command += f"{scale_dir_cov} "
                                            command += f"{overrelaxed_flag} "
                                            command += f"{eigen_method_one_flag} "
                                            command += f"{algo}\n"
                                            f.write(command)

    print(f"wrote file: {filename}")


def funnel_experiment():
    """
    Run funnel experiments

    Stan model: funnel.stan

    Check depth, breadth, and accuracy for 10-D funnel distribution for KLHR algorithms
    """
    filename = "experiments_funnel"

    base = "python experiment_funnel.py"

    wd = "experiments/funnel/"

    prefix = f"#DISBATCH PREFIX ( "
    prefix += f"source experiment_setup ; {base}\n"

    suffix = "#DISBATCH SUFFIX ) &> experiments/${DISBATCH_NAMETASKS}_${DISBATCH_JOBID}_${DISBATCH_TASKID_ZP}.log\n"

    M = 10_000_000
    warmups = [0, M // 2]
    scale_dir_covs = ["-s", ""]
    overrelaxed_flags = ["-o", ""]
    eigen_method_one_flags = ["-e1", ""]
    algos = ["klhr", "klhrsinh"]
    reps = range(10)


    with open(filename, "w") as f:
        f.write(prefix)
        f.write(suffix)
        for algo in algos:
            for r in reps:
                for warmup in warmups:
                    for scale_dir_cov in scale_dir_covs:
                        for overrelaxed_flag in overrelaxed_flags:
                            for eigen_method_one_flag in eigen_method_one_flags:
                                command = f" -M {M} -w {warmup} {r} {scale_dir_cov} {overrelaxed_flag} {eigen_method_one_flag} {algo}\n"
                                f.write(command)

    print(f"wrote file: {filename}")


def relaxation_time_experiment():
    """
    Run relaxation time experiments

    Stan model: earnings.stan

    Explore the effect on relaxation time of the following parameters:

    windowsize -- the size of the initial window in windowed adaptation
    windowscale -- the amount by which to increase the window size after each closed window
    J -- the number of eigenvector to estimate

    """
    filename = "experiments_relaxationtime"

    base = "python experiment_relaxationtime.py -M 30_000 -w 15_000"

    wd = "experiments/relaxationtime/"

    prefix = f"#DISBATCH PREFIX ( "
    prefix += f"source experiment_setup ; {base}\n"

    suffix = "#DISBATCH SUFFIX ) &> experiments/${DISBATCH_NAMETASKS}_${DISBATCH_JOBID}_${DISBATCH_TASKID_ZP}.log\n"

    windowsizes = [25, 50]
    windowscales = [2]
    Js = [2, 3]
    Ls = [0, 2, 4]
    scale_dir_covs = ["-s", ""]
    overrelaxed_flags = ["-o", ""]
    eigen_method_one_flags = ["-e1", ""]
    algos = ["klhr", "klhrsinh"]
    reps = range(10)

    with open(filename, "w") as f:
        f.write(prefix)
        f.write(suffix)
        for wsize in windowsizes:
            for wscale in windowscales:
                for j in Js:
                    for l in Ls:
                        for algo in algos:
                            for r in reps:
                                for scale_dir_cov in scale_dir_covs:
                                    for overrelaxed_flag in overrelaxed_flags:
                                        for eigen_method_one_flag in eigen_method_one_flags:
                                            command = f" --windowsize {wsize} "
                                            command += f"--windowscale {wscale} "
                                            command += f"-J {j} "
                                            command += f"-l {l} "
                                            command += f"-r {r} "
                                            command += f"{scale_dir_cov} "
                                            command += f"{overrelaxed_flag} "
                                            command += f"{eigen_method_one_flag} "
                                            command += f"{algo}\n"
                                            f.write(command)

    print(f"wrote file: {filename}")

if __name__ == "__main__":
    main()
