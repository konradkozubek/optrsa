"""Main module in optrsa project"""
# TODO Read about docstrings and automatic documentation generation services like pydoctor

import cma

import sys
import matplotlib
# PyCharm Professional sets other backend, which may cause trouble
if sys.platform.startswith('darwin'):  # MacOS
    matplotlib.use('MacOSX')
else:
    # Not tested, may need additional dependencies.
    # See https://matplotlib.org/tutorials/introductory/usage.html#backends
    matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

import numpy as np

from typing import Callable, Tuple, Union, List
import abc
import io
import os
import glob
import shutil
import timeit
import subprocess
import multiprocessing.pool
import datetime
import pickle
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl


# Get absolute path to optrsa project directory
_proj_dir = os.path.dirname(__file__)
# Absolute path to python interpreter, assuming that relative path reads /optrsa-py-3-8-1-venv/bin/python
_python_path = _proj_dir + "/optrsa-py-3-8-1-venv/bin/python"  # Or: os.path.abspath("optrsa-py-3-8-1-venv/bin/python")
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
    Current version should be launched only in this module - subprocess wait method should be called at the end of the
    main process, therefore graph_processes global variable should be accessible and wait_for_graphs function should
    be called.
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
                                       + " -code 'N[Area[Region[Apply[RegionUnion,{}]]]]'".format(wolfram_disks_str),
                                       stderr=subprocess.STDOUT, shell=True)
    return float(area_str)


class RSACMAESOptimization(metaclass=abc.ABCMeta):
    """
    Abstract base class for performing optimization of RSA packing fraction with CMA-ES optimizer
    and managing the output data
    """

    # TODO Maybe treat cma_options in the same way as rsa_parameters (default values dictionary, input files)
    default_rsa_parameters: dict = {}
    # Optimization-type-specific rsa parameters - to be set by child classes
    mode_rsa_parameters: dict = {}

    @abc.abstractmethod
    def get_arg_signature(self) -> str:
        return ""

    # TODO Maybe make some attributes obligatory
    def __init__(self,
                 initial_mean: np.ndarray = None,
                 initial_stddevs: float = None,
                 cma_options: dict = None,
                 rsa_parameters: dict = None,
                 accuracy: float = 0.001,
                 parallel: bool = True,
                 input_rel_path: str = None,
                 output_to_file: bool = True,
                 show_graph: bool = False,
                 signature_suffix: str = None) -> None:

        self.initial_mean = initial_mean
        self.initial_stddevs = initial_stddevs
        self.cma_options = cma_options if cma_options is not None else {}
        # Alternative (probably less safe): cma_options or {}
        self.rsa_parameters = rsa_parameters if rsa_parameters is not None else {}
        self.accuracy = accuracy
        self.parallel = parallel
        self.output_to_file = output_to_file
        self.show_graph = show_graph

        # Set optimization signature
        self.signature = datetime.datetime.now().isoformat(timespec="milliseconds")  # Default timezone is right
        self.signature += "-" + type(self).__name__
        self.signature += "-" + self.get_arg_signature()
        self.signature += ("-" + signature_suffix) if signature_suffix is not None else ""
        self.signature = self.signature.replace(":", "-").replace(".", "_")

        # Create output directory
        self.output_dir = _outrsa_dir + "/" + self.signature
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        # Maybe use shutil instead

        # Update self.rsa_parameters by optimization-type-specific parameters
        self.rsa_parameters.update(self.mode_rsa_parameters)

        # Create rsa3d input file in output directory
        if "particleAttributes" in rsa_parameters:
            del rsa_parameters["particleAttributes"]
        self.input_given = input_rel_path is not None
        self.input_filename = _proj_dir + "/" + input_rel_path if self.input_given else None
        # When input file is not given, self.all_rsa_parameters dictionary will be used with self.default_rsa_parameters
        # overwritten by self.rsa_parameters specified in the constructor
        self.all_rsa_parameters = dict(self.default_rsa_parameters, **self.rsa_parameters) if not self.input_given\
            else None
        if self.input_given:
            # Copy input file to output directory and add overwritten options at the end
            # TODO Add checking, if the file exists
            # TODO Test if it works here (worked in optimize_fixed_radii_disks function)
            shutil.copy(self.input_filename, self.output_dir)
            copied_file_name = self.output_dir + "/copied-rsa-input-" + os.path.basename(self.input_filename)
            shutil.move(self.output_dir + "/" + os.path.basename(self.input_filename), copied_file_name)
            with open(copied_file_name, "a") as copied_input_file:
                copied_input_file.write("\n")
                copied_input_file.writelines(["{} = {}\n".format(param_name, param_value)
                                              for param_name, param_value in rsa_parameters.items()])
        else:
            # Create input file in the output directory
            # Use parameters from self.default_rsa_parameters and self.rsa_parameters dictionaries
            with open(self.output_dir + "/generated-rsa-input.txt", "w+") as generated_input_file:
                generated_input_file.writelines(["{} = {}\n".format(param_name, param_value)
                                                 for param_name, param_value in self.all_rsa_parameters.items()])

        # Create file containing output of the rsa3d accuracy mode in output directory
        self.output_filename = self.output_dir + "/packing-fraction-vs-params.txt"
        # Create a file if it does not exist
        with open(self.output_filename, "w+"):  # as output_file
            pass

        # Create list of common arguments (constant during the whole optimization)
        # for running rsa3d program in other process
        self.rsa_proc_arguments = [_rsa_path, "accuracy"]
        if self.input_given:
            # Correct results, but problems with running rsa3d with other command afterwards,
            # because particleAttributes option was overwritten and input file is wrong
            # - the same options have to be given
            self.rsa_proc_arguments.extend(["-f", self.input_filename])
            self.rsa_proc_arguments.extend(["-{}={}".format(param_name, param_value)
                                            for param_name, param_value in self.rsa_parameters.items()])
        else:
            self.rsa_proc_arguments.extend(["-{}={}".format(param_name, param_value)
                                            for param_name, param_value in self.all_rsa_parameters.items()])
        # Index at which particleAttributes parameter will be inserted
        self.rsa_proc_args_last_param_index = len(self.rsa_proc_arguments)
        self.rsa_proc_arguments.extend([str(self.accuracy), self.output_filename])

        # TODO Maybe move these definitions to run method (then redirecting output will be done only in run method and
        #  CMAES initialization information will be added to output file).
        #  Then change here self.CMAES initialization to self.CMAES = None.
        #  In run method, before self.CMAES assignement add printing optimization signature, and after optimization add
        #  printing current time (maybe also the optimization time).
        # Counter of conducted simulations
        self.simulations_num = 0
        # Counter of phenotype candidates in generation
        self.candidate_num = 0
        # Redirect output - done here to catch CMAES initialization information. To be moved to run method together
        # with CMAES initialization
        if self.output_to_file:
            self.stdout = sys.stdout
            self.stderr = sys.stderr
            # If a decorator for redirecting output were used, a "with" statement could have been used
            # TODO Maybe create this file in self.CMAES.logger.name_prefix directory
            # output_file = open(self.output_dir + "/" + self.signature + "_output.txt", "w+")
            output_file = open(self.output_dir + "/optimization-output.txt", "w+")
            sys.stdout = output_file
            sys.stderr = output_file
            # TODO Check, if earlier (more frequent) writing to output file can be forced (something like flush?)
        # Create CMA evolution strategy optimizer object
        # TODO Maybe add serving input files and default values for CMA-ES options (currently directly self.cma_options)
        self.CMAES = cma.CMAEvolutionStrategy(self.initial_mean, self.initial_stddevs,
                                              inopts=self.cma_options)
        self.CMAES.logger.name_prefix += self.signature + "/"  # Default name_prefix is outcmaes/

        # Settings for parallel computation
        # TODO Test if it works in all cases
        # TODO Maybe add ompThreads to common self.rsa_parameters if it will not be changed
        # TODO multiprocessing.cpu_count() instead? (Usage on server)
        self.parallel_threads_number = os.cpu_count()  # * 2
        self.parallel_simulations_number = min(self.parallel_threads_number, self.CMAES.popsize)
        self.omp_threads = self.parallel_threads_number // self.CMAES.popsize\
            if self.parallel and self.parallel_threads_number > self.CMAES.popsize else 1

    def __getstate__(self):
        """
        Method modifying pickling behaviour of the class' instance.
        See https://docs.python.org/3/library/pickle.html#handling-stateful-objects.
        It needs to be overridden by a child class if it defines another unpicklable attributes.
        """

        # Copy the object's state from self.__dict__ which contains all instance attributes using the dict.copy()
        # method to avoid modifying the original state
        state = self.__dict__.copy()
        # Remove the unpicklable entries
        unpicklable_attributes = ["stdout", "stderr"]
        for attr in unpicklable_attributes:
            if attr in state:
                del state[attr]
        return state

    @abc.abstractmethod
    def arg_to_particle_attributes(self, arg: np.ndarray) -> str:
        """Function returning rsa3d program's parameter particleAttributes based on arg"""
        return ""

    # TODO Maybe rename this function (to rsa_simulation_serial) and make the second (rsa_simulation_parallel),
    #  and in current function use nested "with" statements for closing process and file objects after waiting for
    #  the process and return nothing.
    #  Maybe create other functions for reading mean packing fractions from packing-fraction-vs-params.txt file for both
    #  serial and parallel computing cases.
    def get_rsa_simulation_process(self, arg: np.ndarray) -> Tuple[io.TextIOBase, subprocess.Popen]:
        """
        Function running simulations for particle shape specified by arg and returning rsa3d output file
        and rsa3d process
        """

        # Run a process with rsa3d program running simulation
        print("Argument: {}".format(arg))
        rsa_proc_arguments = self.rsa_proc_arguments[:]  # Copy the values of the template arguments
        particle_attributes = self.arg_to_particle_attributes(arg)
        print("particleAttributes: {}".format(particle_attributes))
        rsa_proc_arguments.insert(self.rsa_proc_args_last_param_index, "-particleAttributes=" + particle_attributes)
        # TODO Maybe add setting ompThreads option in case of parallel computation (otherwise specify it explicitly),
        #  maybe calculate ompThreads based on self.CMAES.popsize and number of CPU cores obtained using
        #  multiprocessing module - DONE in rsa_simulation method
        simulation_labels = ",".join([str(self.CMAES.countiter), str(self.candidate_num), str(self.simulations_num)])
        rsa_proc_arguments.append(simulation_labels)
        self.simulations_num += 1
        # TODO In case of reevaluation (UH-CMA-ES), simulation_labels will have to identify the evaluation correctly
        #  (self.simulations_num should be fine to ensure distinction)
        # Create subdirectory for output of rsa3d program in this simulation.
        # simulation_labels contain generation number, candidate number and evaluation number.
        simulation_output_dir = self.output_dir + "/" + simulation_labels.replace(",", "_")
        if not os.path.exists(simulation_output_dir):
            os.makedirs(simulation_output_dir)
        # Maybe use shutil instead
        # Create a file for saving the output of rsa3d program
        # rsa_output_filename = simulation_output_dir + "/packing_" + particle_attributes.replace(" ", "_")
        # # surfaceVolume parameter is not appended to filename if it is given in input file
        # if not self.input_given:
        #     rsa_output_filename += "_" + str(self.all_rsa_parameters["surfaceVolume"])
        # rsa_output_filename += "_output.txt"
        rsa_output_filename = simulation_output_dir + "/rsa-simulation-output.txt"

        # # Serial computing case
        # with open(rsa_output_filename, "w+") as rsa_output_file:
        #     # Open a process with simulation
        #     with subprocess.Popen(rsa_proc_arguments, stdout=rsa_output_file, stderr=rsa_output_file,
        #                           cwd=self.output_dir) as rsa_process:
        #         rsa_process.wait()

        rsa_output_file = open(rsa_output_filename, "w+")
        # TODO To be removed - for debugging
        print(" ".join(rsa_proc_arguments))
        rsa_process = subprocess.Popen(rsa_proc_arguments,
                                       stdout=rsa_output_file,
                                       stderr=rsa_output_file,
                                       cwd=simulation_output_dir)
        return rsa_output_file, rsa_process

    def evaluate_generation_serial(self, pheno_candidates: List[np.ndarray]) -> List[float]:
        self.candidate_num = 0
        values = []
        for pheno_candidate in pheno_candidates:
            print("\nGeneration no. {}, candidate no. {}".format(self.CMAES.countiter, self.candidate_num))

            # Run simulation
            rsa_output_file, rsa_process = self.get_rsa_simulation_process(pheno_candidate)
            rsa_process.wait()
            rsa_output_file.close()
            # TODO Check, if in this solution (without "with" statement) process ends properly
            #  and if this solution is safe
            # Previous, not working solution:
            # with self.rsa_simulation(pheno_candidate) as (rsa_output_file, rsa_process):
            #     rsa_process.wait()

            # Get the packing fraction from the file
            # See https://stackoverflow.com/questions/3346430/what-is-the-most-efficient-way-to-get-first
            # -and-last-line-of-a-text-file
            with open(self.output_filename, "rb") as rsa_output_file:
                # In serial computing reading the last line is sufficient. In parallel computing the right line
                # will have to be found
                rsa_output_file.seek(0, os.SEEK_END)
                while rsa_output_file.read(1) != b"\n":
                    if rsa_output_file.tell() > 1:
                        rsa_output_file.seek(-2, os.SEEK_CUR)
                    else:
                        # Beginning of the file
                        break
                last_line = rsa_output_file.readline().decode()
            # Mean packing fraction is written as the second value, values are separated by tabulators
            mean_packing_fraction = float(last_line.split("\t")[1])
            print("Mean packing fraction: {}".format(mean_packing_fraction))
            values.append(-mean_packing_fraction)
            self.candidate_num += 1
        return values

    # TODO Optimize parallel computing so that all of the time the right (optimal) number of threads is running.
    #  Maybe use multiprocessing.cpu_count() value and multiprocessing.Pool (the latter - if it makes sense in this
    #  case)
    def evaluate_generation_parallel(self, pheno_candidates: List[np.ndarray]) -> List[float]:
        self.candidate_num = 0
        values = np.empty(len(pheno_candidates), dtype=np.float)
        rsa_processes_and_outputs = []
        # TODO Maybe implement repeating these loops if population size is bigger than for example
        #  (number of CPU cores) * 2.
        for pheno_candidate in pheno_candidates:
            rsa_processes_and_outputs.append(self.get_rsa_simulation_process(pheno_candidate))
            self.candidate_num += 1
        # TODO Check if using EvalParallel from cma package or multiprocessing (e.g. Pool) together with Popen
        #  makes sense
        for rsa_output_file, rsa_process in rsa_processes_and_outputs:
            rsa_process.wait()
            rsa_output_file.close()
            print("Generation no. {}, end of candidate no. {} evaluation".format(self.CMAES.countiter,
                                                                                 self.candidate_num))
        with open(self.output_filename, "rb") as rsa_output_file:
            # TODO Maybe find more efficient or elegant solution
            # TODO Maybe iterate through lines in file in reversed order - results of the current generation should be
            #  at the end
            for line in rsa_output_file:
                # Does the line need to be decoded (line_str = line.decode())?
                evaluation_data = line.split(b"\t")
                evaluation_labels = evaluation_data[0].split(b",")
                if int(evaluation_labels[0]) == self.CMAES.countiter:
                    candidate_num = int(evaluation_labels[1])
                    mean_packing_fraction = float(evaluation_data[1])
                    values[candidate_num] = -mean_packing_fraction
        return values.tolist()  # or list(values), because values.ndim == 1

    def rsa_simulation(self, candidate_num: int, arg: np.ndarray) -> None:
        """
        Function running simulations for particle shape specified by arg and waiting for rsa3d process.
        It assigns proper number of OpenMP threads to the rsa3d evaluation.
        """

        # Run a process with rsa3d program running simulation
        rsa_proc_arguments = self.rsa_proc_arguments[:]  # Copy the values of the template arguments
        particle_attributes = self.arg_to_particle_attributes(arg)
        rsa_proc_arguments.insert(self.rsa_proc_args_last_param_index, "-particleAttributes=" + particle_attributes)
        # TODO Maybe move part of this code to constructor
        omp_threads_attribute = str(self.omp_threads)
        if self.parallel and self.parallel_threads_number > self.CMAES.popsize\
                and candidate_num >= self.CMAES.popsize * (self.omp_threads + 1) - self.parallel_threads_number:
            omp_threads_attribute = str(self.omp_threads + 1)
        rsa_proc_arguments.insert(self.rsa_proc_args_last_param_index, "-ompThreads=" + omp_threads_attribute)
        simulation_labels = ",".join([str(self.CMAES.countiter), str(candidate_num), str(self.simulations_num)])
        # Earlier: str(self.simulations_num), str(self.CMAES.countevals), str(self.CMAES.countiter), str(candidate_num)
        # - self.CMAES.countevals value is updated of course only after the end of each generation.
        # self.simulations_num is the number ordering the beginning of simulation, the position of data in
        # packing-fraction-vs-params.txt corresponds to ordering of the end of simulation, and from generation number
        # (self.CMAES.countiter), population size and candidate number one can calculate the number of evaluation
        # mentioned in self.CMAES optimizer e.g. in the result (number of evaluation for the best solution)
        rsa_proc_arguments.append(simulation_labels)
        self.simulations_num += 1
        # TODO In case of reevaluation (UH-CMA-ES), simulation_labels will have to identify the evaluation correctly
        #  (self.simulations_num should be fine to ensure distinction)
        # Create subdirectory for output of rsa3d program in this simulation.
        # simulation_labels contain generation number, candidate number and evaluation number.
        simulation_output_dir = self.output_dir + "/" + simulation_labels.replace(",", "_")
        if not os.path.exists(simulation_output_dir):
            os.makedirs(simulation_output_dir)
            # Maybe use shutil instead
        # Create rsa3d input file containing simulation-specific parameters in simulation output directory
        with open(simulation_output_dir + "/rsa-simulation-input.txt", "w+") as rsa_input_file:
            rsa_input_file.write("ompThreads = {}\n".format(omp_threads_attribute))
            rsa_input_file.write("particleAttributes = {}\n".format(particle_attributes))
        # TODO Maybe problem with wolfram processes will vanish when using two input files (improbable)
        # Create a file for saving the output of rsa3d program
        # rsa_output_filename = simulation_output_dir + "/packing_" + particle_attributes.replace(" ", "_")
        # # surfaceVolume parameter is not appended to filename if it is given in input file
        # if not self.input_given:
        #     rsa_output_filename += "_" + str(self.all_rsa_parameters["surfaceVolume"])
        # rsa_output_filename += "_output.txt"
        rsa_output_filename = simulation_output_dir + "/rsa-simulation-output.txt"

        print("RSA simulation start: generation no. {}, candidate no. {}, simulation no. {}\nArgument: {}\n"
              "particleAttributes: {}".format(*simulation_labels.split(","), arg, particle_attributes))
        # TODO To be removed - for debugging
        # print("RSA simulation process call: {}\n".format(" ".join(rsa_proc_arguments)))
        with open(rsa_output_filename, "w+") as rsa_output_file:
            # Open a process with simulation
            with subprocess.Popen(rsa_proc_arguments,
                                  stdout=rsa_output_file,
                                  stderr=rsa_output_file,
                                  cwd=simulation_output_dir) as rsa_process:
                rsa_process.wait()
        print("RSA simulation end: generation no. {}, candidate no. {}, simulation no. {}".format(
            *simulation_labels.split(",")))

    def evaluate_generation_parallel_in_pool(self, pheno_candidates: List[np.ndarray]) -> List[float]:
        """
        Method running rsa simulations for all phenotype candidates in generation.
        It evaluates self.run_simulation method for proper number of candidates in parallel.
        It uses multiprocessing.pool.ThreadPool for managing a pool of workers.
        concurrent.futures.ThreadPoolExecutor is an alternative with poorer API.
        :param pheno_candidates: List of NumPy ndarrays containing phenotype candidates in generation
        :return: List of fitness function values (minus mean packing fraction) for respective phenotype candidates
        """
        values = np.zeros(len(pheno_candidates), dtype=np.float)
        # It is said that multiprocessing module does not work with class instance method calls,
        # but in this case multiprocessing.pool.ThreadPool seems to work fine with the run_simulation method.
        with multiprocessing.pool.ThreadPool(processes=self.parallel_simulations_number) as pool:
            pool.starmap(self.rsa_simulation, list(enumerate(pheno_candidates)))
            # Maybe read the last packing fraction value after waiting for simulation process in
            # rsa_simulation method and check if the simulation labels are correct (which means that rsa3d program
            # successfully wrote the last line)
            # TODO Maybe add (it works - probably it doesn't need closing the pool):
            # pool.close()
            # pool.join()
        # TODO Consider returning status from Popen objects (if it is possible) and getting them from pool.map and
        #  checking, if rsa simulations finished correctly

        with open(self.output_filename, "rb") as rsa_output_file:
            # TODO Maybe find more efficient or elegant solution
            # TODO Maybe iterate through lines in file in reversed order - results of the current generation should be
            #  at the end
            for line in rsa_output_file:
                # Does the line need to be decoded (line_str = line.decode())?
                evaluation_data = line.split(b"\t")
                evaluation_labels = evaluation_data[0].split(b",")
                if int(evaluation_labels[0]) == self.CMAES.countiter:
                    candidate_num = int(evaluation_labels[1])
                    mean_packing_fraction = float(evaluation_data[1])
                    values[candidate_num] = -mean_packing_fraction
        # TODO Add checking if there exists a zero value in values list and deal with the error (in such a case
        #  record for corresponding candidate wasn't found in packing-fraction-vs-params.txt file)
        return values.tolist()  # or list(values), because values.ndim == 1

    # TODO Create this decorator properly - probably in an inner class or create a parameterized decorator outside the
    #  class - see https://medium.com/@vadimpushtaev/decorator-inside-python-class-1e74d23107f6
    # def redirecting_output(self) -> Callable[[Callable[..., None]], Callable[..., None]]:
    #     if not self.output_to_file:
    #         return lambda f: f
    #
    #     def redirect_output_decorator(function: Callable[..., None]) -> Callable[..., None]:
    #         def redirect_output(*args, **kwargs):
    #             # If self.output_to_file == True, then output will be written to an output file,
    #             # otherwise to the standard output. See http://www.blog.pythonlibrary.org/2016/06/16/python-101
    #             # -redirecting-stdout/
    #             # TODO Maybe use logging module instead, it has useful features.
    #             #  Output can be written simultaneously to both standard output and a file also when using logging
    #             #  module. See: https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and
    #             #  -console-with-scripting
    #             stdout = sys.stdout
    #             stderr = sys.stderr
    #             with open(self.output_dir + "/" + self.signature + "_output.txt", "w+") as output_file:
    #                 sys.stdout = output_file
    #                 sys.stderr = output_file
    #                 function(*args, **kwargs)
    #             sys.stdout = stdout
    #             sys.stderr = stderr
    #         return redirect_output
    #     return redirect_output_decorator

    # TODO Force cma.CMADataLogger to write data to files immediately
    @waiting_for_graphs
    def run(self) -> None:
        """Method running optimization"""

        # If self.output_to_file == True, then output will be written to an output file,
        # otherwise to the standard output. See http://www.blog.pythonlibrary.org/2016/06/16/python-101
        # -redirecting-stdout/
        # TODO Maybe use logging module instead, it has useful features.
        #  Output can be written simultaneously to both standard output and a file also when using logging
        #  module. See: https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and
        #  -console-with-scripting
        # TODO Other option - use the first solution from the above link
        # if self.output_to_file:
        #     stdout = sys.stdout
        #     stderr = sys.stderr
        #     # If a decorator for redirecting output were used, a "with" statement could have been used
        #     # TODO Maybe create this file in self.CMAES.logger.name_prefix directory
        #     # output_file = open(self.output_dir + "/" + self.signature + "_output.txt", "w+")
        #     output_file = open(self.output_dir + "/optimization-output.txt", "w+")
        #     sys.stdout = output_file
        #     sys.stderr = output_file
        #     # TODO Check, if earlier (more frequent) writing to output file can be forced (something like flush?)

        while not self.CMAES.stop():
            print("Generation number {}".format(self.CMAES.countiter))
            pheno_candidates = self.CMAES.ask()
            # TODO Add printing mean candidate solution and standard deviations
            # TODO Maybe in future add plotting an image of a shape corresponding to mean candidate solution
            #  (plotting using matplotlib)
            print("Phenotype candidates:")
            print(pheno_candidates)
            # values = self.evaluate_generation_parallel(pheno_candidates) if self.parallel\
            #     else self.evaluate_generation_serial(pheno_candidates)
            values = self.evaluate_generation_parallel_in_pool(pheno_candidates) if self.parallel\
                else self.evaluate_generation_serial(pheno_candidates)
            # TODO Check, what happens in case when e.g. None is returned as candidate value, so (I guess)
            #  a reevaluation is conducted
            # TODO Maybe add checking if rsa simulation finished with success and successfully wrote a line to
            #  packing-fraction-vs-params.txt file. If it fails, in serial computing the previous packing fraction
            #  is assigned as the current value in values array without any warning, and in parallel - wrong value
            #  from np.empty function is treated as a packing fraction.
            print("End of generation number {}".format(self.CMAES.countiter))
            print("Candidate values:")
            print(values)
            self.CMAES.tell(pheno_candidates, values)
            self.CMAES.logger.add()
            self.CMAES.logger.disp_header()
            self.CMAES.logger.disp([-1])
            print()
        print("\nEnd of optimization\n")
        print(self.CMAES.stop())
        print(self.CMAES.result)
        self.CMAES.result_pretty()

        # Pickling of the object
        # pickle.dump(self, open(self.output_dir + "/_" + self.signature + ".pkl", 'wb'))
        with open(self.output_dir + "/_RSACMAESOptimization.pkl", "wb") as pickle_file:
            pickle.dump(self, pickle_file)
        # Pickling stopped working after switching to Python 3.8.1 (probably it doesn't matter)
        # and moving redirecting output to constructor. After moving redirecting instance variables self.stdout and
        # self.stderr were created. They can't be pickled, which causes following error:
        # "TypeError: cannot pickle '_io.TextIOWrapper' object". It was solved by setting __getstate__ method - pickling
        # and unpickling work correctly now.
        # To unpickle with:
        # rsa_cmaes_optimization = pickle.load(open(self.output_dir + "/_RSACMAESOptimization.pkl", "rb"))
        # It works outside of this module provided that the class of pickled object is imported, e.g.:
        # "from optrsa import FixedRadiiXYPolydiskRSACMAESOpt".

        # TODO Add separate method for making graphs
        # TODO Maybe create another class for analyzing the results of optimization
        if self.show_graph:
            plot_cmaes_graph_in_background(self.CMAES.logger.name_prefix, self.signature)

        # if self.output_to_file:
        #     sys.stdout = stdout
        #     sys.stderr = stderr
        if self.output_to_file:
            sys.stdout = self.stdout
            sys.stderr = self.stderr


class PolydiskRSACMAESOpt(RSACMAESOptimization, metaclass=abc.ABCMeta):

    # mode_rsa_parameters: dict = dict(super().mode_rsa_parameters, particleType="Polydisk")
    # TODO Check, if it is a right way of overriding class attributes (how to get parent class' attribute)
    mode_rsa_parameters: dict = dict(RSACMAESOptimization.mode_rsa_parameters, particleType="Polydisk")

    wolfram_polydisk_area_eval_num: int = -1

    @abc.abstractmethod
    def arg_to_polydisk_attributes(self, arg: np.ndarray) -> Tuple[str, np.ndarray]:
        """
        Function returning part of Polydisk's particleAttributes in a tuple, which first element is \"xy\" or \"rt\"
        string indicating type of coordinates and the second is a numpy ndarray with c01 c02 r0 c11 c12 r1 ... floats
        (disks' coordinates and radii)
        """
        pass

    # TODO Make static?
    def wolfram_polydisk_area(self, disks_params: np.ndarray) -> float:
        """Calculate the area of a polydisk using Wolfram Kernel script"""
        disks_arg = np.reshape(disks_params, (-1, 3))
        wolfram_disks_list = ["Disk[{{{},{}}},{}]".format(*disk) for disk in disks_arg]
        wolfram_disks_str = "{" + ",".join(wolfram_disks_list) + "}"
        wolfram_code = "N[Area[Region[Apply[RegionUnion,{}]]]]".format(wolfram_disks_str)
        # Worked wrong together with ThreadPool parallelization, even when using Python 3.8.1:
        # wolfram_proc_arguments = [_wolfram_path, "-code", wolfram_code]
        # area_str = subprocess.check_output(wolfram_proc_arguments, stderr=subprocess.STDOUT)
        self.wolfram_polydisk_area_eval_num += 1
        output_file_name = self.output_dir + "/wolfram-polydisk-area-" + str(self.wolfram_polydisk_area_eval_num)\
            + ".txt"
        # TODO Check, if output file necessarily has to be created when using system with ">" redirection
        with open(output_file_name, "w+") as output_file:
            pass
        # The solution below works correctly:
        os.system(" ".join([_wolfram_path, "-code", "'" + wolfram_code + "'", ">", output_file_name]))
        area_str = ""
        with open(output_file_name, "r") as output_file:
            for line in output_file:
                area_str += line
        os.remove(output_file_name)
        return float(area_str)

    # TODO Make static?
    def wolframclient_polydisk_area(self, disks_params: np.ndarray) -> float:
        """Calculate the area of a polydisk using wolframclient"""
        disks_arg = np.reshape(disks_params, (-1, 3))
        wl_session = WolframLanguageSession()
        w_disks = [wl.Disk([disk[0], disk[1]], disk[2]) for disk in disks_arg]
        area = wl_session.evaluate(wl.N(wl.Area(wl.Region(wl.Apply(wl.RegionUnion, w_disks)))))
        # A way to export image with polydisk to file:
        # wl_session.evaluate(
        #                 wl.Export("/path/to/file/polydisk.pdf",
        #                 wl.Graphics([wl.Darker(wl.Green, 0.45), w_disks])))
        wl_session.terminate()
        return area

    # @abc.abstractmethod
    # def get_arg_signature(self) -> str:
    #     return ""

    def arg_to_particle_attributes(self, arg: np.ndarray) -> str:
        """Function returning rsa3d program's parameter particleAttributes based on arg"""
        coordinates_type, disks_params = self.arg_to_polydisk_attributes(arg)
        disks_num = disks_params.size // 3
        # area = self.wolfram_polydisk_area(disks_params)
        area = self.wolframclient_polydisk_area(disks_params)
        particle_attributes_list = [str(disks_num), coordinates_type]
        # TODO Maybe do it in a more simple way
        particle_attributes_list.extend(disks_params.astype(np.unicode).tolist())
        particle_attributes_list.append(str(area))
        return " ".join(particle_attributes_list)


class FixedRadiiXYPolydiskRSACMAESOpt(PolydiskRSACMAESOpt):

    default_rsa_parameters = dict(PolydiskRSACMAESOpt.default_rsa_parameters,  # super().default_rsa_parameters,
                                  **{"maxVoxels": "4000000",
                                     "requestedAngularVoxelSize": "0.3",
                                     "minDx": "0.0",
                                     "from": "0",
                                     "collectors": "5",
                                     "split": "100000",
                                     "surfaceDimension": "2",
                                     "boundaryConditions": "periodic"})

    def get_arg_signature(self) -> str:
        disks_num = self.initial_mean.size // 2
        return "disks-" + str(disks_num) + "-initstds-" + str(self.initial_stddevs)

    # TODO Check, if constructor has to be overwritten

    def arg_to_polydisk_attributes(self, arg: np.ndarray) -> Tuple[str, np.ndarray]:
        """
        Function returning part of Polydisk's particleAttributes in a tuple, which first element is \"xy\" or \"rt\"
        string indicating type of coordinates and the second is a numpy ndarray with c01 c02 r0 c11 c12 r1 ... floats
        (disks' coordinates and radii)
        """
        arg_with_radii = np.insert(arg, np.arange(2, arg.size + 1, 2), 1.)
        return "xy", arg_with_radii


@waiting_for_graphs
# TODO Add creating packing_fractions_vs_params.txt at the beginning (currently rsa3d program creates it)
# TODO Add managing errors of the rsa3d program and forcing recalculation (or max n recalculations) or resampling of
#  the parameter point (maybe return None immediately after rsa_process.wait() in case of a failure)
# TODO Maybe create a class representing RSAOptimization and extend it for Polydisks and concrete optimization cases
def optimize_fixed_radii_disks(disks_num: int = 2, initial_stddevs: float = 1.,
                               input_rel_path: str = None,
                               surface_volume: float = 25., accuracy: float = 0.01,
                               output_name_prefix: str = None, store_packings: bool = False) -> None:
    """
    Optimize packing fraction of RSA packings built of unions of intersecting disks with unit radius.
    To be replaced by FixedRadiiXYPolydiskRSACMAESOpt class.
    """

    optimization_signature = datetime.datetime.now().isoformat()\
        + "-fixed-radii-" + str(disks_num) + "-disks-initstds-" + str(initial_stddevs)
    if output_name_prefix is not None:
        optimization_signature += "-" + output_name_prefix
    optimization_signature = optimization_signature.replace(":", "-")
    optimization_signature = optimization_signature.replace(".", "_")
    if input_rel_path is not None:
        input_filename = _proj_dir + "/" + input_rel_path
    output_dir = _outrsa_dir + "/" + optimization_signature
    # Name of the file containing output of the rsa3d accuracy mode
    output_filename = output_dir + "/packing_fractions_vs_params.txt"
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

        # Solution with optional reading input file and overwriting some of the parameters - works, but causes problems
        # with specifying input options afterwards
        print("Argument: {}".format(arg))
        arg_with_radii = np.insert(arg, np.arange(2, arg.size + 1, 2), 1.)
        area = wolfram_polydisk_area(arg_with_radii)
        print("Polydisk area: {}".format(area))
        particle_attributes_list = [str(disks_num), "xy"] + arg_with_radii.astype(np.unicode).tolist() + [str(area)]
        # Create a list of arguments for running rsa3d program in other process
        rsa_proc_arguments = [_rsa_path, "accuracy"]
        if input_rel_path is not None:
            # Correct results, but problems with running rsa3d with other command afterwards,
            # because particleAttributes option was overwritten and input file is wrong
            # - the same options have to be given
            rsa_proc_arguments.extend(["-f", input_filename])
        simulation_parameters["storePackings"] = str(store_packings).lower()
        simulation_parameters["surfaceVolume"] = str(surface_volume)
        simulation_parameters["particleAttributes"] = " ".join(particle_attributes_list)
        rsa_proc_arguments.extend(["-{}={}".format(param_name, param_value)
                                   for param_name, param_value in simulation_parameters.items()])
        rsa_proc_arguments.extend([str(accuracy), output_filename])
        # Open a process with simulation
        rsa_process = subprocess.Popen(rsa_proc_arguments, stderr=subprocess.STDOUT)
        rsa_process.wait()
        # TODO Try parallel computing later (wait for processes after all evaluations
        #  or use EvalParallel from cma package)

        # Move rsa3d output into the output directory
        # TODO Maybe move .bin and .dat files to an output directory subdirectory with a name given by, for example,
        #  evaluation number or [generation number]-[candidate number]
        rsa_output_filenames = glob.glob("packing*")
        for rsa_output_filename in rsa_output_filenames:
            # Seems that sometimes returns error - tries to move a file that has already been moved
            shutil.move(_proj_dir + "/" + rsa_output_filename, output_dir)

        # Get the packing fraction from the file
        # See https://stackoverflow.com/questions/3346430/what-is-the-most-efficient-way-to-get-first-and-last-line
        #  -of-a-text-file
        out_file = open(output_filename, "rb")
        # In serial computing reading the last line is sufficient. In parallel computing the right line
        # will have to be found
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

    # Perform optimization
    # TODO Maybe separate CMA-ES console output from RSA console output
    # TODO Print to a file, rather than to the standard output
    # cma.CMAOptions().pprint()  # Print available options
    evol_strat = cma.CMAEvolutionStrategy(np.zeros(2 * disks_num), initial_stddevs,
                                          inopts={"maxiter": 10, "verb_disp": 1, "verbose": 4, "seed": 1234})
    # "maxfevals": 30
    evol_strat.logger.name_prefix += optimization_signature + "/"  # Default name_prefix is outcmaes/
    while not evol_strat.stop():
        pheno_candidates = evol_strat.ask()
        print("Generation number {}".format(evol_strat.countiter))
        # TODO Add printing mean candidate solution and standard deviations
        # TODO Maybe in future add plotting an image of a shape corresponding to mean candidate solution
        #  (plotting using matplotlib)
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
        print("End of generation number {}".format(evol_strat.countiter))
        print("Candidate values:")
        print(values)
        evol_strat.tell(pheno_candidates, values)
        evol_strat.logger.add()
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


@waiting_for_graphs
def optimize_three_fixed_radii_disks(initial_stddevs: float = 1.,
                                     input_rel_path: str = None,
                                     surface_volume: float = 80., accuracy: float = 0.01,
                                     output_name_prefix: str = None, store_packings: bool = False) -> None:
    """
    Optimize packing fraction of RSA packings built of unions of intersecting disks with unit radius.
    To be replaced by Disks3FixedRadiiXYPolydiskRSACMAESOpt class.
    """

    optimization_signature = datetime.datetime.now().isoformat() + "-three-fixed-radii-disks-initstds-" + str(initial_stddevs)  # Default timezone is right
    if output_name_prefix is not None:
        optimization_signature += "-" + output_name_prefix
    optimization_signature = optimization_signature.replace(":", "-")
    optimization_signature = optimization_signature.replace(".", "_")
    if input_rel_path is not None:
        input_filename = _proj_dir + "/" + input_rel_path
    output_dir = _outrsa_dir + "/" + optimization_signature
    # Name of the file containing output of the rsa3d accuracy mode
    output_filename = output_dir + "/packing_fractions_vs_params.txt"
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
        arg_with_radii = np.array([arg[0], arg[1], 1., arg[2], 0., 1., 0., 0., 1.])
        area = wolfram_polydisk_area(arg_with_radii)
        print("Polydisk area: {}".format(area))
        particle_attributes_list = ["3", "xy"] + arg_with_radii.astype(np.unicode).tolist() + [str(area)]
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
        rsa_process = subprocess.Popen(rsa_proc_arguments, stderr=subprocess.STDOUT, cwd=output_dir)
        rsa_process.wait()  # TODO Try parallel computing later (wait for processes after all evaluations or use EvalParallel from cma package)

        # # Move rsa3d output into the output directory
        # # TODO Maybe move .bin and .dat files to an output directory subdirectory with a name given by, for example, evaluation number or [generation number]-[candidate number]
        # rsa_output_filenames = glob.glob("packing*")
        # for rsa_output_filename in rsa_output_filenames:
        #     # Seems that sometimes returns error - tries to move a file that has already been moved
        #     shutil.move(_proj_dir + "/" + rsa_output_filename, output_dir)

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

    # Perform optimization
    # TODO Maybe separate CMA-ES console output from RSA console output
    # TODO Print to a file, rather than to the standard output
    # cma.CMAOptions().pprint()  # Print available options
    evol_strat = cma.CMAEvolutionStrategy(np.zeros(3), initial_stddevs, inopts={"maxiter": 10, "popsize": 10, "verb_disp": 1, "verbose": 4, "seed": 1234})  # "maxfevals": 30
    evol_strat.logger.name_prefix += optimization_signature + "/"  # Default name_prefix is outcmaes/
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
        print("End of generation number {}".format(evol_strat.countiter))
        print("Candidate values:")
        print(values)
        evol_strat.tell(pheno_candidates, values)
        evol_strat.logger.add()
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


# TODO Maybe use argparse module
# TODO Maybe prepare a Makefile like in https://docs.python-guide.org/writing/structure/ and with creating
#  virtual environment (check how PyCharm creates virtualenvs)
# TODO Object-oriented implementation of optimization - class RSACMAESOptimization, - partly DONE
#  maybe class PolydiskRSACMAESOpt(RSACMAESOptimization), - DONE
#  __init__ arguments: mean, stds, RSAParameters, CMAOptions, other parameters, - DONE (there is no separate cl. RSAP.)
#  separate setting RSA parameters (kept constant during the whole optimization) from objective_function, - DONE
#  separate setting particleAttributes from the rest of objective_function (to arg_to_particle_attributes method), -DONE
#  parallel version of objective_function, find an efficient way to get packing fraction and make a method for this,DONE
#  change simulation_parameters, evol_strat and other variables to class attributes, - DONE
#  maybe move input file generation to __init__ - DONE
#  (in __init__: setting packing signature, creating packing-fraction-vs-params.txt, - DONE
#  setting (also default) RSAParameters, generating input file, preparing rsa3d process call sequence
#  (apart from particleAttributes it is constant during the optimization)), - DONE
#  separate method for running optimization called run (or execute or optimize - the latter one can be confused
#  with evol_strat.optimize), - DONE
#  methods for plotting and visualization of the saved data
#  then program mode with RSAParameters instantiation, calling run method and maybe other actions like plotting graphs
#  or making wolfram files etc. (respective methods).
#  The objective is to make running, analyzing and modification of optimization comfortable.
# TODO Maybe add a method calling rsa3d program in wolfram mode on specified .bin file
#  - particleAttributes may be taken from rsa-simulation-input.txt file
# TODO Adding box constraints based on feasible anisotropy (probably)
# TODO Does writing to the single packing-fraction-vs-params.txt file by paralleled processes pose a problem?
# TODO Maybe decreasing packing fraction error in subsequent generations by accuracy mode
#  or increasing the number of collectors. Maybe combine that with decreasing the population.
# TODO Algorithms for noisy optimization: UH-CMA-ES (cma.NoiseHandler)? DX-NES-UE?
# TODO Maybe single, big collectors and uncertainty handling (variable numbers of reevaluations,
#  thus variable numbers of collectors - UH-CMA-ES)?
# TODO Use random seed generation in rsa3d program
# TODO Add making correct graphs
# TODO Test on server
# TODO Check if the optimization works correctly
# TODO Does storing data affect performance?
# TODO Think, if demanding given accuracy (rsa3d accuracy mode) is the right thing to do
# TODO Check, if rsa3d options are well chosen (especially split) and wonder, if they should be somehow
#  automatically adjusted during optimization
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
            optimize_fixed_radii_disks(disks_num=int(sys.argv[2]), input_rel_path=input_rel_path,
                                       store_packings=bool(int(sys.argv[4])))
        else:
            optimize_fixed_radii_disks(disks_num=int(sys.argv[2]), initial_stddevs=float(sys.argv[5]),
                                       input_rel_path=input_rel_path, store_packings=bool(int(sys.argv[4])))
    elif sys.argv[1] == "optthreefixedradii":
        if sys.argv[4] != "None":
            input_rel_path = sys.argv[4]
        else:
            input_rel_path = None
        optimize_three_fixed_radii_disks(initial_stddevs=float(sys.argv[2]), input_rel_path=input_rel_path,
                                         store_packings=bool(int(sys.argv[3])))
    elif sys.argv[1] == "optfixedradiiclass":
        disks_num = int(sys.argv[2])
        initial_stddevs = float(sys.argv[3])
        if sys.argv[4] != "None":
            input_rel_path = sys.argv[4]
        else:
            input_rel_path = None
        initial_mean = np.zeros(2 * disks_num)
        fixed_radii_polydisk_opt = FixedRadiiXYPolydiskRSACMAESOpt(initial_mean=initial_mean,
                                                                   initial_stddevs=initial_stddevs,
                                                                   cma_options={"maxiter": 2,  # 10
                                                                                "verb_disp": 1,
                                                                                "verbose": 4,
                                                                                "popsize": 3
                                                                                },
                                                                   rsa_parameters={"surfaceVolume": "500.",  #50.
                                                                                   "storePackings": "true"},
                                                                   accuracy=0.01,  # Debugging
                                                                   parallel=True,
                                                                   output_to_file=True,
                                                                   # show_graph=True,
                                                                   signature_suffix="test-3-pool-parallel")
        fixed_radii_polydisk_opt.run()
