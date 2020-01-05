"""Main module in optrsa project"""
# TODO Read about docstrings and automatic documentation generation services like pydoctor

import cma

import sys
import matplotlib
# PyCharm Professional sets other backend, which may cause trouble
if sys.platform.startswith('darwin'): # MacOS
    matplotlib.use('MacOSX')
else:
    matplotlib.use("Qt5Agg") # Not tested, may need additional dependencies. See https://matplotlib.org/tutorials/introductory/usage.html#backends
import matplotlib.pyplot as plt

import numpy as np

from typing import Callable
import os
import glob
import shutil
import timeit
import subprocess
import datetime


# Get absolute path to optrsa project directory
_proj_dir = os.path.dirname(__file__)
# Absolute path to python interpreter, assuming that relative path reads /virtualenv/bin/python
_python_path = _proj_dir + "/virtualenv/bin/python" # Or: os.path.abspath("virtualenv/bin/python")
# Absolute path to Wolfram Kernel script, assuming that relative path reads /exec/wolframscript
_wolfram_path = _proj_dir + "/exec/wolframscript"
# Absolute path to rsa3d executable compiled with target 2.1, assuming that relative path reads /exec/rsa.2.1
_rsa_path = _proj_dir + "/exec/rsa.2.1"
# Absolute paths to input and output directories
_input_dir = _proj_dir + "/input"
_outrsa_dir = _proj_dir + "/outrsa"
_outcmaes_dir = _proj_dir + "/outcmaes"
graph_processes = []


def test_cma_package() -> None:
    """Tests cma package using doctest, according to http://cma.gforge.inria.fr/apidocs-pycma/cma.html"""
    print("Testing time (tests should run without complaints in about between 20 and 100 seconds):",
          timeit.timeit(stmt='cma.test.main()', setup="import cma.test", number=1))


# TODO Maybe write a class for managing graph processes
def plot_cmaes_graph_in_background(data_dir: str, window_name: str) -> None:
    """
    Plot CMA-ES data in a different process.
    Window will be shown without blocking execution and will not be closed when the main process ends working.
    Current version should be launched only in this module - subprocess wait method should be called at the end of the main process,
    therefore graph_processes global variable should be accessible and wait_for_graphs function should be called.
    """
    graph_process = subprocess.Popen([_python_path, _proj_dir + "/plot_cmaes_data.py", data_dir, window_name])
    graph_processes.append(graph_process)


def wait_for_graphs() -> None:
    for plot_process in graph_processes:
        plot_process.wait()


def waiting_for_graphs(function: Callable[..., None]) -> Callable[..., None]:
    def wrapped(*args, **kwargs) -> None:
        function(*args, **kwargs)
        wait_for_graphs()
    return wrapped


@waiting_for_graphs
def example_cma_plots() -> None:
    """
    Illustrative CMA-ES data plotting in subprocesses.
    Example with optimization of a simple function.
    See http://cma.gforge.inria.fr/apidocs-pycma/cma.evolution_strategy.CMAEvolutionStrategy.html
    """
    es_first = cma.CMAEvolutionStrategy(4 * [0.2], 0.5, {'verb_disp': 0})
    es_first.logger.name_prefix += "sphere-test-1/"
    es_first.logger.disp_header()  # annotate the print of disp
    while not es_first.stop():
        X = es_first.ask()
        es_first.tell(X, [cma.ff.sphere(x) for x in X])
        es_first.logger.add()  # log current iteration
        es_first.logger.disp([-1])  # display info for last iteration
    es_first.logger.disp_header()

    plot_cmaes_graph_in_background(es_first.logger.name_prefix, "Sphere 1")

    print()
    es_second = cma.CMAEvolutionStrategy(4 * [0.2], 0.5, {'verb_disp': 0})
    es_second.logger.name_prefix += "sphere-test-2/"
    es_second.logger.disp_header()
    while not es_second.stop():
        X = es_second.ask()
        es_second.tell(X, [cma.ff.sphere(x) for x in X])
        es_second.logger.add()
        es_second.logger.disp([-1])
    es_second.logger.disp_header()

    plot_cmaes_graph_in_background(es_second.logger.name_prefix, "Sphere 2")


def wolfram_polydisk_area(arg: np.ndarray) -> float:
    """Calculate the area of a polydisk using Wolfram Kernel script"""
    disks_arg = np.reshape(arg, (-1, 3))
    wolfram_disks_list = ["Disk[{{{},{}}},{}]".format(*disk) for disk in disks_arg]
    wolfram_disks_str = "{" + ",".join(wolfram_disks_list) + "}"
    # TODO Check, if the Wolfram Kernel script can be called with shell=False and whether the performance will be better
    area_str = subprocess.check_output(_wolfram_path
                                       + " -code 'Area[Region[Apply[RegionUnion,{}]]]'".format(wolfram_disks_str),
                                       stderr=subprocess.STDOUT, shell=True)
    return float(area_str)


@waiting_for_graphs
def optimize_fixed_radii_disks(disks_num: int = 2, initial_stddevs: float = 1.,
                               input_rel_path: str = None,
                               surface_volume: float = 25., accuracy: float = 0.01,
                               output_name_prefix: str = None, store_packings: bool = False) -> None:
    """Optimize packing fraction of RSA packings built of unions of intersecting disks with unit radius"""

    optimization_signature = datetime.datetime.now().isoformat() + "-fixed-radii-" + str(disks_num) + "-disks-initstds-" + str(initial_stddevs)  # Default timezone is right
    if output_name_prefix is not None:
        optimization_signature += "-" + output_name_prefix
    optimization_signature = optimization_signature.replace(":", "-")
    optimization_signature = optimization_signature.replace(".", "_")
    if input_rel_path is not None:
        input_filename = _proj_dir + "/" + input_rel_path
    output_dir = _outrsa_dir + "/" + optimization_signature
    # Name of the file containing output of the rsa3d accuracy mode
    output_filename = output_dir + "/accuracies.txt"
    # Create output directory, if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Simulation parameters that are the same during the whole optimization
    simulation_parameters = {}
    if input_rel_path is None:
        simulation_parameters.update({"maxVoxels": "4000000",
                                      "requestedAngularVoxelSize": "0.3",
                                      "minDx": "0.0",
                                      "from": "0",
                                      "collectors": "5",
                                      "split": "100000",
                                      "surfaceDimension": "2",
                                      "boundaryConditions": "periodic",
                                      "particleType": "Polydisk"})

    def objective_function(arg: np.ndarray) -> float:
        """Function running simulations for particle shape specified by arg and returning mean packing fraction"""

        # Run a process with rsa3d program running simulation

        # Solution with optional reading input file and overwriting some of the parameters - works, but causes problems with specifying input options afterwards
        print("Argument: {}".format(arg))
        arg_with_radii = np.insert(arg, np.arange(2, arg.size + 1, 2), 1.)
        area = wolfram_polydisk_area(arg_with_radii)
        print("Polydisk area: {}".format(area))
        particle_attributes_list = [str(disks_num), "xy"] + arg_with_radii.astype(np.unicode).tolist() + [str(area)]
        # Create a list of arguments for running rsa3d program in other process
        rsa_proc_arguments = [_rsa_path, "accuracy"]
        if input_rel_path is not None:
            # Correct results, but problems with running rsa3d with other command afterwards, because particleAttributes option was overwritten and input file is wrong - the same options have to be given
            rsa_proc_arguments.extend(["-f", input_filename])
        simulation_parameters["storePackings"] = str(store_packings).lower()
        simulation_parameters["surfaceVolume"] = str(surface_volume)
        simulation_parameters["particleAttributes"] = " ".join(particle_attributes_list)
        rsa_proc_arguments.extend(["-{}={}".format(param_name, param_value) for param_name, param_value in simulation_parameters.items()])
        rsa_proc_arguments.extend([str(accuracy), output_filename])
        # Open a process with simulation
        rsa_process = subprocess.Popen(rsa_proc_arguments, stderr=subprocess.STDOUT)
        rsa_process.wait()  # TODO Try parallel computing later (wait for processes after all evaluations or use EvalParallel from cma package)

        # Move rsa3d output into the output directory
        # TODO Maybe move .bin and .dat files to an output directory subdirectory with a name given by, for example, iteration number or [generation number]-[iteration number]
        rsa_output_filenames = glob.glob("packing*")
        for rsa_output_filename in rsa_output_filenames:
            # Seems that sometimes returns error - tries to move a file that has already been moved
            shutil.move(_proj_dir + "/" + rsa_output_filename, output_dir)

        # Get the packing fraction from the file
        # See https://stackoverflow.com/questions/3346430/what-is-the-most-efficient-way-to-get-first-and-last-line-of-a-text-file
        out_file = open(output_filename, "rb")
        # In serial computing reading the last line is sufficient. In parallel computing the right line will have to be found
        out_file.seek(0, os.SEEK_END)
        while out_file.read(1) != b"\n":
            if out_file.tell() > 1:
                out_file.seek(-2, os.SEEK_CUR)
            else:
                # Beginning of the file
                break
        last_line = out_file.readline().decode()
        out_file.close()
        # Mean packing fraction is written as the first value in line and separated by a tabulator
        mean_packing_fraction = float(last_line.partition("\t")[0])
        print("Mean packing fraction: {}".format(mean_packing_fraction))
        return mean_packing_fraction

    # print(objective_function(np.array([0., 0., 0.5, 0, 0.25, 0.5])))
    # print(objective_function(np.array([0., 0., 0.5, 0, 0.5, 0.5])))

    # Perform optimization
    # TODO Maybe separate CMA-ES console output from RSA console output
    # TODO Print to a file, rather than to the standard output
    # cma.CMAOptions().pprint()  # Print available options
    evol_strat = cma.CMAEvolutionStrategy(np.zeros(2 * disks_num), initial_stddevs, inopts={"maxiter": 10, "verb_disp": 1, "verbose": 4, "seed": 1234})  # "maxfevals": 30
    evol_strat.logger.name_prefix += optimization_signature + "/"  # Default name_prefix is outcmaes/
    evol_strat.logger.disp_header()
    while not evol_strat.stop():
        pheno_candidates = evol_strat.ask()
        print("Generation number {}".format(evol_strat.countiter))
        # TODO Add printing mean candidate solution and standard deviations
        # TODO Maybe in future add plotting an image of a shape corresponding to mean candidate solution (plotting using matplotlib)
        print("Phenotype candidates:")
        print(pheno_candidates)
        values = []
        candidate_number = 1
        for pheno_candidate in pheno_candidates:
            # TODO Check, what evol_strat.countevals really means.
            print("\nGeneration nr {}, candidate nr {}".format(evol_strat.countiter, candidate_number))
            value = objective_function(pheno_candidate)
            print("\nGeneration nr {}, candidate nr {}".format(evol_strat.countiter, candidate_number))
            print("Returned value: {}".format(value))
            values.append(-value)
            candidate_number += 1
        # values = [-objective_function(pheno_candidate) for pheno_candidate in pheno_candidates]
        evol_strat.tell(pheno_candidates, values)
        evol_strat.logger.add()
        print("End of generation number {}".format(evol_strat.countiter))
        evol_strat.logger.disp_header()
        evol_strat.logger.disp([-1])
        print()
    print("\nEnd of optimization\n")
    print(evol_strat.stop())
    print(evol_strat.result)
    evol_strat.result_pretty()
    # For some shapes rsa3d terminates with error message:
    # terminate called after throwing an instance of 'ValidationException'
    #   what():  Packing linear size is <= 2 neighbour list cell size - boundary conditions will break.
    # Probably distance between two disks is to big compared to packing size
    # TODO Check, whether forcing objective_function to return None is such situations is a right thing to do

    if input_rel_path is not None:
        # Copy input file to output directory and add overwritten options at the end, excluding particleAttributes
        shutil.copy(input_filename, output_dir)
        copied_file_name = output_dir + "/copied-input-" + os.path.basename(input_filename)
        shutil.move(output_dir + "/" + os.path.basename(input_filename), copied_file_name)
        copied_input_file = open(copied_file_name, "a")
        copied_input_file.write("\n")
        copied_input_file.writelines(["{} = {}\n".format(param_name, simulation_parameters[param_name])
                                      for param_name in ["storePackings", "surfaceVolume"]])
        copied_input_file.close()
    else:
        # Create input file in the output directory
        # Use parameters from simulation_parameters dictionary, excluding particleAttributes
        written_simulation_params = simulation_parameters
        del written_simulation_params["particleAttributes"]
        generated_input_file = open(output_dir + "/generated-input.txt", "w+")
        generated_input_file.writelines(["{} = {}\n".format(param_name, param_value)
                                         for param_name, param_value in written_simulation_params.items()])
        generated_input_file.close()

    # TODO Add saving graph to file
    plot_cmaes_graph_in_background(evol_strat.logger.name_prefix, "Two disks fixed radii")


# TODO Maybe add a function calling rsa3d program with in wolfram mode on specified .bin file - particleAttributes have to be extracted and passed
# TODO Maybe use argparse module
# TODO Maybe prepare a Makefile like in https://docs.python-guide.org/writing/structure/ and with creating virtual environment (check how PyCharm creates virtualenvs)
# TODO Think, if demanding given accuracy (rsa3d accuracy mode) is the right thing to do
# TODO Check, if rsa3d options are well chosen (especially split) and wonder, if they should be somehow automatically adjusted during optimization
# TODO Does storing data affect performance?
if __name__ == '__main__':
    if sys.argv[1] == "testcma":
        test_cma_package()
    elif sys.argv[1] == "examplecmaplots":
        example_cma_plots()
    elif sys.argv[1] == "optfixedradii":
        if sys.argv[3] != "None":
            input_rel_path = sys.argv[3]
        else:
            input_rel_path = None
        # TODO Rearrange arguments and delete if - solution below is to guarantee backward compatibility
        if len(sys.argv) < 6:
            optimize_fixed_radii_disks(disks_num=int(sys.argv[2]), input_rel_path=input_rel_path, store_packings=bool(int(sys.argv[4])))
        else:
            optimize_fixed_radii_disks(disks_num=int(sys.argv[2]), initial_stddevs=float(sys.argv[5]), input_rel_path=input_rel_path, store_packings=bool(int(sys.argv[4])))
