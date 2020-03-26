import optrsa
import argparse
import subprocess
import glob
import pandas as pd
import sys
import logging
import os
import multiprocessing
from typing import NamedTuple

opt_data_columns = ["generationnum", "candidatenum", "simulationnum",
                    "partattrs", "pfrac", "pfracstddev", "collectorsnum"]


def load_optimization_data(signature: str) -> pd.DataFrame:
    optimization_output_dir = optrsa._output_dir + "/" + signature
    output_filename = optimization_output_dir + "/packing-fraction-vs-params.txt"
    with open(output_filename) as output_file:
        # optimization_data = pd.DataFrame(opt_data_columns=opt_data_columns)
        simulation_data_list = []
        for line in output_file:
            evaluation_data = line.rstrip("\n").split("\t")
            evaluation_labels = evaluation_data[0].split(",")
            simulation_output_dir = optimization_output_dir + "/outrsa/" + "_".join(evaluation_labels)
            # Get collectors' number
            rsa_data_file_lines_count = subprocess.check_output(["wc", "-l",
                                                                 glob.glob(simulation_output_dir + "/*.dat")[0]])
            collectors_num = int(rsa_data_file_lines_count.strip().split()[0])
            # If multiple lines in packing-fraction-vs-params.txt file correspond to the same candidate, the
            # values from the last such line will be used
            # optimization_data = optimization_data.append(dict(zip(opt_data_columns, [*map(int, evaluation_labels),
            #                                                                 float(evaluation_data[1]),
            #                                                                 float(evaluation_data[2]),
            #                                                                 collectors_num])), ignore_index=True)
            simulation_data_list.append(pd.DataFrame([[*map(int, evaluation_labels),
                                                       evaluation_data[4],
                                                       float(evaluation_data[1]),
                                                       float(evaluation_data[2]),
                                                       collectors_num]], columns=opt_data_columns))
    optimization_data = pd.concat(simulation_data_list, ignore_index=True)
    optimization_data.set_index(opt_data_columns[:3], drop=False, inplace=True)
    return optimization_data


if __name__ == '__main__':
    module_description = "Repeating specified simulations from an optimization" \
                         " implemented in a wild way in order to make use of the optimization classes' methods."
    arg_parser = argparse.ArgumentParser(description=module_description)
    arg_parser.add_argument("-s", "--signature", help="optimization signature - name of subdirectory of ./output")
    arg_parser.add_argument("-c", "--condition", help="condition under which to repeat simulation"
                                                      " written in the form of Python code using following"
                                                      " optimization data column names as variables: "
                                                      + ", ".join(opt_data_columns))
    arg_parser.add_argument("-t", "--threads", help="number of threads to use")
    args = arg_parser.parse_args()
    if args.signature is None:
        raise TypeError("Optimization signature has to be specified using -s argument")
    if args.condition is None:
        raise TypeError("Condition has to be specified using -c argument")

    # Get optimization object
    opt_class_name = args.signature.split("-")[5]
    # Get optimization class from optrsa module.
    opt_class = getattr(sys.modules["optrsa"], opt_class_name)
    exec("from optrsa import {}".format(opt_class_name))
    optimization = opt_class.unpickle(args.signature)
    repeated_simulations_dir = optimization.output_dir + "/repeated-simulations"
    if not os.path.exists(repeated_simulations_dir):
        os.makedirs(repeated_simulations_dir)
    # Prepare the logger
    logging_handler = logging.FileHandler(filename=repeated_simulations_dir + "/simulations-output.log", mode="a")
    logging_handler.setFormatter(optimization.logger.handlers[0].formatter)
    for handler in optimization.logger.handlers[:]:
        optimization.logger.removeHandler(handler)
    optimization.logger.addHandler(logging_handler)

    # Load optimization data and choose simulations specified by the condition
    optimization_data = load_optimization_data(args.signature)
    optimization_data.sort_values(by="pfrac", ascending=False, inplace=True)
    condition = args.condition
    for column_name in opt_data_columns:
        condition = condition.replace(column_name, "optimization_data[\"{}\"]".format(column_name))
    optimization.logger.info(msg="Optimization data:")
    optimization.logger.info(msg=optimization_data)
    simulations_data = optimization_data.loc[eval(condition)]
    # simulations_data.sort_values(by="pfrac", ascending=False, inplace=True)

    # Repeat simulations
    optimization.logger.info(msg="Repeating simulations satisfying the condition: {}".format(args.condition))
    optimization.logger.info(msg="Simulations' data:")
    optimization.logger.info(msg=simulations_data)
    # Prepare optimization object
    optimization.output_filename = repeated_simulations_dir + "/packing-fraction-vs-params.txt"
    optimization.parallel_threads_number = int(args.threads) if args.threads is not None else os.cpu_count()  # * 2
    optimization.parallel_simulations_number = min(optimization.parallel_threads_number, simulations_data.shape[0])
    optimization.pool_workers_number = optimization.parallel_simulations_number
    optimization.remaining_pool_simulations = simulations_data.shape[0]
    optimization.rsa_processes_stdins = {}
    optimization.rsa_output_dir = repeated_simulations_dir + "/outrsa"
    if not os.path.exists(optimization.rsa_output_dir):
        os.makedirs(optimization.rsa_output_dir)
    optimization.rsa_proc_arguments[3] = glob.glob(optimization.output_dir + "/copied-rsa-input-*")[0]
    optimization.rsa_proc_arguments[-1] = optimization.output_filename

    def repeat_simulation(simulation_data: NamedTuple, omp_threads: int) -> None:
        optimization.CMAES.countiter = simulation_data.Index[0]
        optimization.simulations_num = simulation_data.Index[2]
        optimization.rsa_simulation(simulation_data.Index[1], None, simulation_data.partattrs, omp_threads)

    # Repeat simulations parallel in pool
    with multiprocessing.pool.ThreadPool(processes=optimization.pool_workers_number) as pool:
        omp_threads_numbers = [optimization.omp_threads_number(num, optimization.pool_workers_number,
                                                                    optimization.parallel_threads_number)
                               for num in range(simulations_data.shape[0])]
        simulations_arguments = list(zip(list(simulations_data.itertuples(name="Simulation")), omp_threads_numbers))
        pool.starmap(repeat_simulation, simulations_arguments)
