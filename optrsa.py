"""Main module in optrsa project"""
# TODO Read about docstrings and automatic documentation generation services like pydoctor

import cma

import sys
import matplotlib
import subprocess
# PyCharm Professional sets other backend, which may cause trouble
if sys.platform.startswith('darwin'):  # MacOS
    # See https://matplotlib.org/tutorials/introductory/usage.html#backends
    matplotlib.use("MacOSX")
    # Qt5Agg, Qt4Agg - can't import qt bindings, GTK3Agg, GTK3Cairo - can't install all dependencies. nbAgg - fails.
    # WX does not have access to the screen. TkAgg works, WebAgg (with tornado imported) works worse than TkAgg.
else:
    try:
        # TODO Maybe use platform module instead
        system_info = subprocess.check_output(["uname", "-mrs"]).decode().strip()
    except Exception as exception:
        system_info = "not checked"
    okeanos_system_info = "Linux 4.12.14-150.17_5.0.86-cray_ari_s x86_64"
    if system_info != okeanos_system_info and system_info != "not checked":
        # Partially tested
        matplotlib.use("Qt5Agg")  # Maybe try also TkAgg (works) if interactivity is needed. Agg is not interactive.
import matplotlib.pyplot as plt
import matplotlib.transforms
import matplotlib.offsetbox
import matplotlib.patches
import matplotlib.text
# import matplotlib_shiftable_annotation

import numpy as np
from scipy.spatial import ConvexHull
import shapely.geometry

import argparse
import json
import pandas as pd
from typing import Callable, Tuple, Union, List, Optional
from collections import namedtuple
import abc
import inspect
import io
import os
import glob
import shutil
from file_read_backwards import FileReadBackwards
import traceback
import logging
import logging.config
import yaml
import pprint
import timeit
# import subprocess
import multiprocessing.pool
# TODO Maybe import MPIPoolExecutor only in the optimize mode, if okeanos_parallel option is set
from mpi4py.futures import MPIPoolExecutor
from concurrent.futures import Future
import threading
import time
import datetime
import pickle
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl


# Get absolute path to optrsa project directory
_proj_dir = os.path.dirname(__file__)
# TODO Maybe adjust it in order to point to the correct virtual environment
# Absolute path to python interpreter, assuming that relative path reads /optrsa-py-3-8-1-venv/bin/python
_python_path = _proj_dir + "/optrsa-py-3-8-1-venv/bin/python"  # Or: os.path.abspath("optrsa-py-3-8-1-venv/bin/python")
# Absolute path to Wolfram Kernel script, assuming that relative path reads /exec/wolframscript
_wolfram_path = _proj_dir + "/exec/wolframscript"
# Absolute path to rsa3d executable compiled with target 2.1, assuming that relative path reads /exec/rsa.2.1
_rsa_path = _proj_dir + "/exec/rsa.2.1"
# Absolute paths to input and output directories
_input_dir = _proj_dir + "/input"
_output_dir = _proj_dir + "/output"
_outrsa_dir = _proj_dir + "/outrsa"  # To be removed
_outcmaes_dir = _proj_dir + "/outcmaes"  # To be removed
_cmaes_logging_config = _proj_dir + "/optimization_logging.yaml"
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


def softplus(x, k: Optional[float] = 1):
    """
    Calculate element-wise softplus of the NumPy array_like x argument with "sharpness" parameter k

    :param x: Argument suitable to pass to NumPy functions using array_like arguments
    :param k: Optional parameter determining "sharpness" of the function, defaults to 1
    :return: Object of the same type as x, element-wise softplus of x
    """
    return np.log(1 + np.exp(k * x)) / k


def logistic(x, min: Optional[float] = 0, max: Optional[float] = 1, k: Optional[float] = 1):
    """
    Calculate element-wise logistic function of the NumPy array_like x argument with "sharpness" parameter k

    :param x: Argument suitable to pass to NumPy functions using array_like arguments
    :param min: Optional parameter defining left asymptote of the function, defaults to 0
    :param max: Optional parameter defining right asymptote of the function, defaults to 1
    :param k: Optional parameter determining "steepness" of the function, defaults to 1
    :return: Object of the same type as x, element-wise logistic function of x
    """
    return min + (max - min) / (1 + np.exp(-k * x))


class StreamToLogger:
    """
    Source: https://stackoverflow.com/questions/11124093/redirect-python-print-output-to-logger/11124247
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ""
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == "\n":
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != "":
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ""


class OptimizationFormatter(logging.Formatter):
    """
    Partly inspired by https://stackoverflow.com/questions/18639387/how-change-python-logging-to-display-time-passed
    -from-when-the-script-execution
    """
    def format(self, record):
        # It seems that this method may be called multiple times when processing a log record, so modifying record's
        # attributes based on their previous values (e.g. extending a string) won't work as expected (can be done
        # multiple times).
        # Add attribute with readable running time (time since logging module was loaded)
        time_diff = str(datetime.timedelta(milliseconds=record.relativeCreated))
        record.runningTime = time_diff[:-3]
        # return super(OptimizationFormatter, self).format(record)

        # Dealing with logging messages with multiple lines
        # Record before calling format has only msg attribute, message attribute is created in format method
        # To illustrate it (and the fact that the format method is called multiple times) uncomment:
        # with open("test.txt", "a") as file:
        #     file.write(record.msg + " " + str(hasattr(record, "message")) + "\n")
        message = record.getMessage()
        if "\n" not in message or record.exc_info or record.exc_text or record.stack_info:
            # If the message doesn't contain newlines or contains exception or stack information, print in a standard
            # way without dealing with newlines
            return super(OptimizationFormatter, self).format(record)
        else:
            msg = record.msg
            record.msg = ""
            prefix = super(OptimizationFormatter, self).format(record)
            record.msg = msg
            # indentation = " " * len(prefix)
            # message_lines = message.split("\n")
            # output = prefix + message_lines.pop(0) + "\n"
            # for line in message_lines:
            #     output += indentation + line + "\n"
            message_lines = message.split("\n")
            output = ""
            for line in message_lines:
                output += prefix + line + "\n"
            return output[:-1]


# TODO Maybe change it into a typed named tuple
DefaultRSASimulationResult = namedtuple(typename="DefaultRSASimulationResult",
                                        field_names=["candidate_num", "simulation_num", "first_collector_num",
                                                     "collectors_num", "return_code", "node_message", "pid",
                                                     "start_time", "time", "particles_numbers"])


class RSACMAESOptimization(metaclass=abc.ABCMeta):
    """
    Abstract base class for performing optimization of RSA packing fraction with CMA-ES optimizer
    and managing the output data
    """

    # TODO Maybe treat cma_options in the same way as rsa_parameters (default values dictionary, input files)
    default_rsa_parameters: dict = {}
    # Optimization-type-specific rsa parameters - to be set by child classes
    mode_rsa_parameters: dict = {}

    # TODO Add names of the used files and directories as class attributes
    # rsa_output_dirname = "outrsa"
    # cmaes_output_dirname = "outcmaes"
    # opt_input_filename = "optimization-input.json"
    # gen_rsa_input_filename = "generated-rsa-input.txt"
    # cp_rsa_input_filename_prefix = "copied-rsa-input-"
    # rsa_sim_input_filename = "rsa-simulation-input.txt"
    # rsa_sim_output_filename = "rsa-simulation-output.txt"
    # output_filename = "packing-fraction-vs-params.dat"
    # opt_output_filename = "optimization-output.txt"
    # opt_data_output_filename = "optimization.dat"

    optimization_data_columns: dict = {"generationnum": np.int,
                                       "meanarg": str,
                                       "meanpartattrs": str,
                                       "stddevs": str,
                                       "covmat": str,
                                       "partstddevs": str,
                                       "bestind": np.int, "bestarg": str, "bestpartattrs": str,
                                       "bestpfrac": np.float, "bestpfracstddev": np.float,
                                       "medianind": np.int, "medianarg": str, "medianpartattrs": str,
                                       "medianpfrac": np.float, "medianpfracstddev": np.float,
                                       "worstind": np.int, "worstarg": str, "worstpartattrs": str,
                                       "worstpfrac": np.float, "worstpfracstddev": np.float,
                                       "candidatesdata": str}

    # DefaultRSASimulationResult = namedtuple(typename="DefaultRSASimulationResult",
    #                                         field_names=["candidate_num", "simulation_num", "first_collector_num",
    #                                                      "collectors_num", "return_code", "node_message", "pid",
    #                                                      "start_time", "time", "particles_numbers"])

    stddevs_sample_size_optclattr: int = None

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
                 threads: int = None,
                 particle_attributes_parallel: bool = False,
                 okeanos: bool = False,
                 max_nodes_number: int = None,
                 okeanos_parallel: bool = False,
                 nodes_number: int = None,
                 min_collectors_number: int = 10,
                 collectors_per_task: int = 1,
                 input_rel_path: str = None,
                 output_to_file: bool = True,
                 output_to_stdout: bool = False,
                 log_generations: bool = True,
                 show_graph: bool = False,
                 signature_suffix: str = None,
                 optimization_input: dict = None,
                 # Keyword arguments are captured to receive unnecessarily passed, ignored optimization class attributes
                 **kwargs
                 ) -> None:

        self.initial_mean = initial_mean
        self.initial_stddevs = initial_stddevs
        self.cma_options = cma_options if cma_options is not None else {}
        # Alternative (probably less safe): cma_options or {}
        self.rsa_parameters = rsa_parameters if rsa_parameters is not None else {}
        self.accuracy = accuracy
        self.parallel = parallel
        self.particle_attributes_parallel = particle_attributes_parallel
        self.okeanos = okeanos
        self.max_nodes_number = max_nodes_number
        self.okeanos_parallel = okeanos_parallel
        self.nodes_number = nodes_number
        self.min_collectors_number = max(min_collectors_number, 2)
        self.collectors_per_task = collectors_per_task
        self.output_to_file = output_to_file
        self.output_to_stdout = output_to_stdout
        self.log_generations = log_generations
        self.show_graph = show_graph
        self.optimization_input = optimization_input

        self.set_optimization_class_attributes(optimization_input=self.optimization_input)

        # Set optimization signature
        self.signature = datetime.datetime.now().isoformat(timespec="milliseconds")  # Default timezone is right
        self.signature += "-" + type(self).__name__
        self.signature += "-" + self.get_arg_signature()
        self.signature += ("-" + signature_suffix) if signature_suffix is not None else ""
        self.signature = self.signature.replace(":", "-").replace(".", "_")

        # Create output directory and subdirectories for RSA and CMAES output
        self.output_dir = _output_dir + "/" + self.signature
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.rsa_output_dir = self.output_dir + "/outrsa"
        if not os.path.exists(self.rsa_output_dir):
            os.makedirs(self.rsa_output_dir)
        self.cmaes_output_dir = self.output_dir + "/outcmaes"
        if not os.path.exists(self.cmaes_output_dir):
            os.makedirs(self.cmaes_output_dir)
        # Maybe use shutil instead

        # Generate used optimization input file in output directory
        with open(self.output_dir + "/optimization-input.json", "w+") as opt_input_file:
            json.dump(self.optimization_input, opt_input_file, indent=2)

        # Update self.rsa_parameters by optimization-type-specific parameters
        self.rsa_parameters.update(self.mode_rsa_parameters)

        # Create rsa3d input file in output directory
        if "particleAttributes" in rsa_parameters:
            del rsa_parameters["particleAttributes"]
        self.input_given = input_rel_path is not None
        self.input_filename = _input_dir + "/" + input_rel_path if self.input_given else None
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
        self.set_rsa_proc_arguments()

        # Configure and set optimization state logger
        # Logs can be printed to a logfile or to the standard output. By default, logfile will contain log records of
        # severity level at least logging.INFO and standard output - at least logging.DEBUG.
        with open(_cmaes_logging_config) as config_file:
            logging_configuration = yaml.full_load(config_file)
        if self.output_to_file:
            # To set handlers' filename, modify configuration dictionary as below, try to set it to a variable in
            # configuration file, use logger's addHandler method or modify logger.handlers[0] (probably it is not
            # possible to specify file name after handler was instatiated)
            logging_configuration["handlers"]["optimization_logfile"]["filename"] = self.output_dir\
                + "/optimization-output.log"
        else:
            logging_configuration["loggers"]["optrsa.optimization"]["handlers"].pop(0)
            del logging_configuration["handlers"]["optimization_logfile"]
        if not self.output_to_stdout:
            logging_configuration["loggers"]["optrsa.optimization"]["propagate"] = False
        logging.config.dictConfig(logging_configuration)
        self.logger = logging.getLogger("optrsa.optimization")
        # Maybe optionally add a handler to print everything to the console
        # TODO Maybe move these definitions to run method (then redirecting output will be done only in run method and
        #  CMAES initialization information will be added to output file).
        #  Then change here self.CMAES initialization to self.CMAES = None.
        #  In run method, before self.CMAES assignment add printing optimization signature, and after optimization add
        #  printing current time (maybe also the optimization time).
        # Counter of conducted simulations
        self.simulations_num = 0
        # Counter of phenotype candidates in generation
        self.candidate_num = 0
        # Redirect output - done here to catch CMAES initialization information. To be moved to run method together
        # with CMAES initialization
        # if self.output_to_file:
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        # If a decorator for redirecting output were used, a "with" statement could have been used
        # TODO Maybe create this file in self.CMAES.logger.name_prefix directory
        # output_file = open(self.output_dir + "/" + self.signature + "_output.txt", "w+")
        # output_file = open(self.output_dir + "/optimization-output.log", "w+")
        sys.stdout = StreamToLogger(logger=self.logger, log_level=logging.INFO)
        sys.stderr = StreamToLogger(logger=self.logger, log_level=logging.ERROR)
        # TODO Check, if earlier (more frequent) writing to output file can be forced (something like flush?)
        # Create CMA evolution strategy optimizer object
        # TODO Maybe add serving input files and default values for CMA-ES options (currently directly self.cma_options)
        self.logger.info(msg="Optimization class: {}".format(self.__class__.__name__))
        self.logger.info(msg="Optimizer:")
        self.CMAES = cma.CMAEvolutionStrategy(self.initial_mean, self.initial_stddevs,
                                              inopts=self.cma_options)
        # self.CMAES.logger.name_prefix += self.signature + "/"  # Default name_prefix is outcmaes/
        self.CMAES.logger.name_prefix = self.cmaes_output_dir + "/"

        # Settings for parallel computation
        # TODO Maybe add ompThreads to common self.rsa_parameters if it will not be changed
        # TODO multiprocessing.cpu_count() instead? (Usage on server)
        self.parallel_threads_number = threads if threads is not None else os.cpu_count()  # * 2
        if not self.okeanos:
            self.parallel_simulations_number = min(self.parallel_threads_number, self.CMAES.popsize)
        else:
            self.parallel_simulations_number = min(self.max_nodes_number - 1, self.CMAES.popsize) \
                if self.max_nodes_number is not None else self.CMAES.popsize
        # self.omp_threads = self.parallel_threads_number // self.CMAES.popsize\
        #     if self.parallel and self.parallel_threads_number > self.CMAES.popsize else 1
        # if self.okeanos:
        #     from mpi4py.futures import MPIPoolExecutor
        if self.okeanos_parallel and self.nodes_number is None:
            # It is assumed that the SLURM job has the same number of assigned nodes
            self.nodes_number = 1 + self.CMAES.popsize

        # Create file for logging generation data
        self.opt_data_filename = self.output_dir + "/optimization.dat" if self.log_generations else None
        if self.log_generations:
            # Create a file if it does not exist
            # with open(self.opt_data_filename, "w+"):
            #     pass
            with open(self.opt_data_filename, "w+") as opt_data_file:
                # Write header line
                opt_data_file.write("\t".join(self.optimization_data_columns) + "\n")

    def set_rsa_proc_arguments(self) -> None:
        """
        Create list of common arguments (constant during the whole optimization)
        for running rsa3d program in other process
        :return: None
        """
        self.rsa_proc_arguments = [_rsa_path]
        # TODO Maybe implement in in another way
        if not self.okeanos_parallel:
            self.rsa_proc_arguments.append("accuracy")
        else:
            self.rsa_proc_arguments.append("simulate")
        if self.input_given:
            self.rsa_proc_arguments.extend(["-f", self.input_filename])
            self.rsa_proc_arguments.extend(["-{}={}".format(param_name, param_value)
                                            for param_name, param_value in self.rsa_parameters.items()])
        else:
            self.rsa_proc_arguments.extend(["-{}={}".format(param_name, param_value)
                                            for param_name, param_value in self.all_rsa_parameters.items()])
        if not self.okeanos_parallel:
            # Index at which particleAttributes parameter will be inserted
            self.rsa_proc_args_last_param_index = len(self.rsa_proc_arguments)
            self.rsa_proc_arguments.extend([str(self.accuracy), self.output_filename])

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
        unpicklable_attributes = ["stdout", "stderr", "logger", "rsa_processes_stdins"]
        for attr in unpicklable_attributes:
            if attr in state:
                del state[attr]
        return state

    def __setstate__(self, state):
        """
        Method modifying unpickling behaviour of the class' instance.
        See https://docs.python.org/3/library/pickle.html#handling-stateful-objects.
        It needs to be overridden by a child class if it defines another unpicklable attributes.
        """

        # Restore pickled instance attributes
        self.__dict__.update(state)

        # Restore unpicklable attributes
        # Configure and set optimization state logger
        # TODO Check, if the file handler always appends to the file and doesn't truncate it at the beginning
        with open(_cmaes_logging_config) as config_file:
            logging_configuration = yaml.full_load(config_file)
        if self.output_to_file:
            # To set handlers' filename, modify configuration dictionary as below, try to set it to a variable in
            # configuration file, use logger's addHandler method or modify logger.handlers[0] (probably it is not
            # possible to specify file name after handler was instantiated)
            logging_configuration["handlers"]["optimization_logfile"]["filename"] = self.output_dir \
                                                                                    + "/optimization-output.log"
        else:
            logging_configuration["loggers"]["optrsa.optimization"]["handlers"].pop(0)
            del logging_configuration["handlers"]["optimization_logfile"]
        if not self.output_to_stdout:
            logging_configuration["loggers"]["optrsa.optimization"]["propagate"] = False
        logging.config.dictConfig(logging_configuration)
        self.logger = logging.getLogger("optrsa.optimization")
        # Redirect output
        # TODO Maybe separate redirecting output from unpickling in order to be able to unpickle and use standard output
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = StreamToLogger(logger=self.logger, log_level=logging.INFO)
        sys.stderr = StreamToLogger(logger=self.logger, log_level=logging.ERROR)

    def pickle(self, name: Optional[str] = None) -> None:
        pickle_name = "" if name is None else "-" + name
        with open(self.output_dir + "/_" + self.__class__.__name__ + pickle_name + ".pkl", "wb") as pickle_file:
            pickle.dump(self, pickle_file)

    # TODO Annotate it correctly
    @classmethod
    def unpickle(cls, signature: str):
        with open(_output_dir + "/" + signature + "/_" + cls.__name__ + ".pkl", "rb") as pickle_file:
            return pickle.load(pickle_file)
        # Unpickling works outside of this module provided that the class of pickled object is imported, e.g.:
        # "from optrsa import FixedRadiiXYPolydiskRSACMAESOpt".
        # TODO Maybe add a method or program mode to unpickle optimization object, change CMA-ES options such as
        #  termination condition and resume optimization - probably run method will work fine

    @classmethod
    def set_optimization_class_attributes(cls, signature: Optional[str] = None,
                                          optimization_input: Optional[dict] = None):
        if optimization_input is None:
            # Get optimization input data from optimization directory
            with open(_output_dir + "/" + signature + "/optimization-input.json", "r") as opt_input_file:
                # TODO Maybe use configparser module or YAML format instead
                optimization_input = json.load(opt_input_file)

        # Set optimization class attibutes (class attributes with suffix "_optclattr") using optimization input data
        # Based on https://www.geeksforgeeks.org/how-to-get-a-list-of-class-attributes-in-python/
        suffix = "_optclattr"
        # Get pairs (member, value) for current optimization class
        for member in inspect.getmembers(cls):
            if member[0].endswith(suffix) and not (member[0].startswith("_") or inspect.ismethod(member[1])):
                attr_name = member[0][:-len(suffix)]
                setattr(cls, member[0], optimization_input["opt_class_args"][attr_name])

    @classmethod
    @abc.abstractmethod
    def arg_to_particle_attributes(cls, arg: np.ndarray) -> str:
        """Function returning rsa3d program's parameter particleAttributes based on arg"""
        return ""

    @classmethod
    def arg_to_particle_parameters(cls, arg: np.ndarray) -> np.ndarray:
        """Function returning particle's parameters based on arg"""
        return arg

    @classmethod
    def arg_in_domain(cls, arg: np.ndarray) -> bool:
        """Function checking if arg belongs to the optimization domain"""
        return True

    @classmethod
    def swap_arg(cls, arg: np.ndarray) -> np.ndarray:
        """
        Function swapping arg to another, for which the objective function value is the same. In some cases it may be
        useful to do it in order to manage plateaus. By default, it does not change the argument.
        """
        return arg

    @classmethod
    @abc.abstractmethod
    def stddevs_to_particle_stddevs(cls, arg: np.ndarray, stddevs: np.ndarray, covariance_matrix: np.ndarray) \
            -> np.ndarray:
        """
        Function returning particle's parameters' standard deviations based on standard deviations (and possibly mean
        coordinates) in optimization's space
        """
        # TODO Check, how standard deviations should be transformed
        optimization_sample = np.random.default_rng().multivariate_normal(mean=arg,
                                                                          cov=covariance_matrix,
                                                                          size=cls.stddevs_sample_size_optclattr)
        particle_parameters_sample = np.apply_along_axis(func1d=cls.arg_to_particle_parameters,
                                                         axis=1,
                                                         arr=optimization_sample)
        particle_parameters_stddevs = np.std(particle_parameters_sample, axis=0, dtype=np.float64)
        return particle_parameters_stddevs

    @classmethod
    @abc.abstractmethod
    def draw_particle(cls, particle_attributes: str, scaling_factor: float, color: str,
                      arg: Optional[np.ndarray] = None, std_devs: Optional[np.ndarray] = None,
                      covariance_matrix: Optional[np.ndarray] = None, part_std_devs: Optional[np.ndarray] = None) \
            -> matplotlib.offsetbox.DrawingArea:
        """
        Abstract class method drawing particle described by `particle_attributes` string attribute on
        matplotlib.offsetbox.DrawingArea and returning DrawingArea object.

        :param particle_attributes: Particle's particleAttributes rsa3d program's parameter string
        :param scaling_factor: Factor for scaling objects drawn on matplotlib.offsetbox.DrawingArea
        :param color: Particle's color specified by matplotlib's color string
        :param arg: Argument point describing the particle - may be given if it is needed to draw the particle with full
                    information
        :param std_devs: Standard deviations of the probability distribution - may be given if it is needed to draw the
                         particle corresponding to the mean of the probability distribution
        :param covariance_matrix: Covariance matrix of the probability distribution - may be given if it is needed to
                                  draw the particle corresponding to the mean of the probability distribution
        :param part_std_devs: Particle attributes' standard deviations - they may be given in order to show them on the
                              drawing of the particle corresponding to the mean of the probability distribution
        :return: matplotlib.offsetbox.DrawingArea object with drawn particle
        """
        pass

    def omp_threads_number(self, simulation_number: int, parallel_simulations_number: int,
                           parallel_threads_number: int) -> int:
        """
        Method calculating number of OpenMP threads to assign to rsa3d program's process.

        :param simulation_number: Number (index) of simulation from range [0, parallel_simulations_number - 1]
        :param parallel_simulations_number: Number of parallel running simulations
        :param parallel_threads_number: Overall available number of threads to assign to simulations
        :return: Number of OpenMP threads to assign to rsa3d program's process
        """
        if parallel_threads_number < parallel_simulations_number:
            self.logger.warning(msg="Assigned threads number {} is lower than"
                                    " parallelized simulations number {}".format(parallel_threads_number,
                                                                                 parallel_simulations_number))
        if not self.parallel or parallel_threads_number <= parallel_simulations_number:
            return 1
        omp_threads = self.parallel_threads_number // parallel_simulations_number
        if simulation_number >= parallel_simulations_number * (omp_threads + 1) - self.parallel_threads_number:
            omp_threads += 1
        return omp_threads

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
        simulation_labels = ",".join([str(self.CMAES.countiter), str(self.candidate_num), str(self.simulations_num),
                                      " ".join(map(str, arg))])
        rsa_proc_arguments.append(simulation_labels)
        self.simulations_num += 1
        # TODO In case of reevaluation (UH-CMA-ES), simulation_labels will have to identify the evaluation correctly
        #  (self.simulations_num should be fine to ensure distinction)
        # Create subdirectory for output of rsa3d program in this simulation.
        # simulation_labels contain generation number, candidate number and evaluation number.
        simulation_output_dir = self.rsa_output_dir + "/" + "_".join(simulation_labels.split(",")[:3])
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
        #                           cwd=self.rsa_output_dir) as rsa_process:
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

    def rsa_simulation(self, candidate_num: int, arg: np.ndarray,
                       particle_attributes: Optional[str] = None, omp_threads: Optional[int] = None,
                       simulation_num: Optional[int] = None) -> int:
        """
        Function running simulations for particle shape specified by arg and waiting for rsa3d process.
        It assigns proper number of OpenMP threads to the rsa3d evaluation.

        :param candidate_num: Identification number of the candidate
        :param arg: Phenotype candidate's point
        :param particle_attributes: Particle attributes computed for arg - if not given, they are computed
        :param omp_threads: Number of OpenMP threads to assign to rsa3d program - if not given, they are computed
        :param simulation_num: Simulation number - needs to be passed only when self.okeanos option is True and the
                               method is called by evaluate_generation_parallel_in_pool method
        :return: RSA simulation's return code
        """

        try:
            sim_start_time = datetime.datetime.now()
            # Run a process with rsa3d program running simulation
            rsa_proc_arguments = self.rsa_proc_arguments[:]  # Copy the values of the template arguments
            if particle_attributes is None:
                particle_attributes = self.arg_to_particle_attributes(arg)
            rsa_proc_arguments.insert(self.rsa_proc_args_last_param_index, "-particleAttributes=" + particle_attributes)
            # TODO Maybe move part of this code to constructor
            # omp_threads_attribute = str(self.omp_threads)
            # if self.parallel and self.parallel_threads_number > self.CMAES.popsize\
            #         and candidate_num >= self.CMAES.popsize * (self.omp_threads + 1) - self.parallel_threads_number:
            #     omp_threads_attribute = str(self.omp_threads + 1)
            if omp_threads is not None:
                omp_threads_attribute = str(omp_threads)
            else:
                if self.parallel:
                    if self.okeanos:
                        omp_threads_attribute = str(self.parallel_threads_number)
                    else:
                        omp_threads_attribute = str(self.omp_threads_number(candidate_num, self.pool_workers_number,
                                                                            self.parallel_threads_number))
                else:
                    omp_threads_attribute = str(self.parallel_threads_number)
            rsa_proc_arguments.insert(self.rsa_proc_args_last_param_index, "-ompThreads=" + omp_threads_attribute)
            # Maybe use candidate_num instead of simulation_num to label rsa3d processes' stdins
            okeanos_node_process = simulation_num is not None
            if self.parallel and not okeanos_node_process:
                simulation_num = self.simulations_num
            simulation_labels = ",".join([str(self.CMAES.countiter), str(candidate_num),
                                          str(simulation_num if self.parallel else self.simulations_num),
                                          " ".join(map(str, arg))])
            # Earlier: str(self.simulations_num), str(self.CMAES.countevals), str(self.CMAES.countiter),
            # str(candidate_num) - self.CMAES.countevals value is updated of course only after the end of each
            # generation. self.simulations_num is the number ordering the beginning of simulation, the position of data
            # in packing-fraction-vs-params.txt corresponds to ordering of the end of simulation, and from generation
            # number (self.CMAES.countiter), population size and candidate number one can calculate the number of
            # evaluation mentioned in self.CMAES optimizer e.g. in the result (number of evaluation for the best
            # solution)
            rsa_proc_arguments.append(simulation_labels)
            if not (self.okeanos and okeanos_node_process):
                # If this method was run by MPIPoolExecutor worker, original optimization object would not be modified
                self.simulations_num += 1
            # TODO In case of reevaluation (UH-CMA-ES), simulation_labels will have to identify the evaluation correctly
            #  (self.simulations_num should be fine to ensure distinction)
            # Create subdirectory for output of rsa3d program in this simulation.
            # simulation_labels contain generation number, candidate number and evaluation number.
            simulation_output_dir = self.rsa_output_dir + "/" + "_".join(simulation_labels.split(",")[:3])
            if not os.path.exists(simulation_output_dir):
                os.makedirs(simulation_output_dir)
                # Maybe use shutil instead
            # Create rsa3d input file containing simulation-specific parameters in simulation output directory
            with open(simulation_output_dir + "/rsa-simulation-input.txt", "w+") as rsa_input_file:
                rsa_input_file.write("ompThreads = {}\n".format(omp_threads_attribute))
                rsa_input_file.write("particleAttributes = {}\n".format(particle_attributes))
            # Check the node ID if run on Okeanos
            node_message = ""
            if self.okeanos:
                try:
                    node_id = subprocess.check_output(["hostname"]).decode().strip()
                except subprocess.CalledProcessError as exception:
                    self.logger.warning(msg="subprocess.CalledProcessError raised when checking host name:"
                                            " command: {}, return code: {}, output: \"{}\".\n{}\n"
                                            "Node ID will not be logged.".format(exception.cmd,
                                                                                 -exception.returncode,
                                                                                 exception.output.decode(),
                                                                                 traceback.format_exc(limit=6).strip()))
                    node_id = "not checked"
                except Exception as exception:
                    self.logger.warning(msg="Exception raised when checking host name; {}: {}\n"
                                            "{}".format(type(exception).__name__, exception,
                                                        traceback.format_exc(limit=6).strip()))
                    node_id = "not checked"
                node_message = "NID: {}, ".format(node_id)
            # Create a file for saving the output of rsa3d program
            rsa_output_filename = simulation_output_dir + "/rsa-simulation-output.txt"
            with open(rsa_output_filename, "w+") as rsa_output_file:
                # Open a process with simulation
                with subprocess.Popen(rsa_proc_arguments,
                                      stdin=subprocess.PIPE,  # Maybe specify it only if self.parallel
                                      stdout=rsa_output_file,
                                      stderr=rsa_output_file,
                                      cwd=simulation_output_dir) as rsa_process:
                    pid = rsa_process.pid
                    self.logger.info(msg="RSA simulation start: generation no. {}, candidate no. {}, simulation no. {},"
                                         " {}PID: {}, ompThreads: {}\n"
                                         "Argument: {}\n"
                                         "particleAttributes: {}".format(*simulation_labels.split(",")[:3],
                                                                         node_message,
                                                                         pid,
                                                                         omp_threads_attribute,
                                                                         pprint.pformat(arg),
                                                                         particle_attributes))
                    # For debugging
                    # self.logger.debug(msg="RSA simulation process call: {}".format(" ".join(rsa_proc_arguments)))
                    if self.parallel and not self.okeanos:
                        self.rsa_processes_stdins[simulation_num] = rsa_process.stdin
                    return_code = rsa_process.wait()
            sim_end_time = datetime.datetime.now()
            threads_message = ""
            if self.parallel and not self.okeanos:
                self.remaining_pool_simulations -= 1
                del self.rsa_processes_stdins[simulation_num]
                if 0 < self.remaining_pool_simulations < self.pool_workers_number:
                    # Send messages with new numbers of OpenMP threads to rsa3d processes
                    # TODO Maybe remember numbers of threads and send messages only when numbers need to be changed
                    threads_message = "\nRemaining parallel simulations: {}." \
                                      " Increasing numbers of OpenMP threads.\n" \
                                      "simulation number: ompThreads".format(self.remaining_pool_simulations)
                    for rsa_process_num, remaining_sim_num, rsa_process_stdin\
                            in zip(list(range(len(self.rsa_processes_stdins))),
                                   list(self.rsa_processes_stdins),
                                   list(self.rsa_processes_stdins.values())):
                        new_omp_threads = self.omp_threads_number(rsa_process_num, self.remaining_pool_simulations,
                                                                  self.parallel_threads_number)
                        # Flushing is needed to send the message and the newline is needed for rsa3d program to end
                        # reading the message
                        rsa_process_stdin.write("ompThreads:{}\n".format(new_omp_threads).encode())
                        rsa_process_stdin.flush()
                        threads_message += "\n{}: {}".format(remaining_sim_num, new_omp_threads)
            # Get collectors' number
            # rsa_data_file_lines_count = subprocess.check_output(["wc", "-l",
            #                                                      glob.glob(simulation_output_dir + "/*.dat")[0]])
            # On Okeanos in the first generation this command used to fail
            try:
                rsa_data_file_lines_count = subprocess.check_output(["wc", "-l",
                                                                     glob.glob(simulation_output_dir + "/*.dat")[0]])
                collectors_num_message = str(int(rsa_data_file_lines_count.strip().split()[0]))
            except subprocess.CalledProcessError as exception:
                self.logger.warning(msg="subprocess.CalledProcessError raised when checking collectors number:"
                                        " command: {}, return code: {},"
                                        " output: \"{}\".\n{}".format(exception.cmd,
                                                                      -exception.returncode,
                                                                      exception.output.decode(),
                                                                      traceback.format_exc(limit=6).strip()))
                try:
                    rsa_data_file_lines_count = subprocess.check_output(["wc", "-l",
                                                                         glob.glob(simulation_output_dir
                                                                                   + "/*.dat")[0]],
                                                                        shell=True)
                    collectors_num_message = str(int(rsa_data_file_lines_count.strip().split()[0]))
                except subprocess.CalledProcessError as exception:
                    self.logger.warning(msg="subprocess.CalledProcessError raised when checking collectors number with"
                                            " \"shell\" option set to True: command: {}, return code: {},"
                                            " output: \"{}\".\n{}\nCollectors number"
                                            " will not be logged.".format(exception.cmd,
                                                                          -exception.returncode,
                                                                          exception.output.decode(),
                                                                          traceback.format_exc(limit=6).strip()))
                    collectors_num_message = "not checked"
                except Exception as exception:
                    self.logger.warning(msg="Exception raised when checking collectors number; {}: {}\n{}\nCollectors"
                                            " number will not be logged.".format(type(exception).__name__, exception,
                                                                                 traceback.format_exc(limit=6).strip()))
                    collectors_num_message = "not checked"
            except Exception as exception:
                self.logger.warning(msg="Exception raised when checking collectors number; {}: {}\n{}\nCollectors"
                                        " number will not be logged.".format(type(exception).__name__, exception,
                                                                             traceback.format_exc(limit=6).strip()))
                collectors_num_message = "not checked"
            self.logger.info(msg="RSA simulation end: generation no. {}, candidate no. {}, simulation no. {},"
                                 " {}PID: {}. Time: {}, collectors: {}, return code: {}"
                                 "{}".format(*simulation_labels.split(",")[:3],
                                             node_message,
                                             pid,
                                             str(sim_end_time - sim_start_time),
                                             collectors_num_message,
                                             str(return_code),
                                             threads_message))
            # self.logger.debug(msg="remaining_pool_simulations: {}, rsa_processes_stdins number: {}".format(
            #     self.remaining_pool_simulations, len(self.rsa_processes_stdins)))
            # self.logger.debug(msg="rsa_processes_stdins:\n{}".format(self.rsa_processes_stdins))
        # TODO Maybe check also for subprocess.CalledProcessError and set return code appropriately
        except Exception as exception:
            self.logger.warning(msg="Exception raised in rsa_simulation method for generation no. {}, candidate no. {},"
                                    " argument: {}; {}: {}\n"
                                    "{}\n"
                                    "Candidate will be resampled.".format(str(self.CMAES.countiter), str(candidate_num),
                                                                          pprint.pformat(arg),
                                                                          type(exception).__name__, exception,
                                                                          traceback.format_exc(limit=6).strip()))
            return_code = -1
        return return_code

    # def safe_rsa_simulation_wrapper(self, rsa_simulation_method: Callable[..., int]) -> Callable[..., int]:
    #     def safe_rsa_simulation(custom_self, candidate_num: int, arg: np.ndarray,
    #                             particle_attributes: Optional[str] = None, omp_threads: Optional[int] = None,
    #                             simulation_num: Optional[int] = None) -> int:
    #         rsa_simulation_method.__doc__
    #
    #         try:
    #             return_code = rsa_simulation_method(candidate_num, arg, particle_attributes,
    #                                                 omp_threads, simulation_num)
    #         # TODO Maybe check also for subprocess.CalledProcessError and set return code appropriately
    #         except Exception as exception:
    #             self.logger.warning(msg="Exception raised in rsa_simulation method for generation no. {},"
    #                                     " candidate no. {}, simulation no. {}; {}: {}\n{}\n"
    #                                     "Candidate will be resampled.".format(self.CMAES.countiter, candidate_num,
    #                                                                           simulation_num,
    #                                                                           type(exception).__name__, exception,
    #                                                                           traceback.format_exc(limit=6).strip()))
    #             return_code = -1
    #         return return_code
    #     return safe_rsa_simulation

    # TODO Maybe find a way to pass arguments to optrsa program's process in runtime to change the overall number of
    #  used threads from the next generation on
    def evaluate_generation_parallel_in_pool(self, pheno_candidates: List[np.ndarray],
                                             cand_particle_attributes: Optional[List[str]] = None)\
            -> Tuple[List[float], List[int]]:
        """
        Method running rsa simulations for all phenotype candidates in generation.
        It evaluates self.run_simulation method for proper number of candidates in parallel.
        It uses multiprocessing.pool.ThreadPool for managing a pool of workers.
        concurrent.futures.ThreadPoolExecutor is an alternative with poorer API.

        :param pheno_candidates: List of NumPy ndarrays containing phenotype candidates in generation
        :param cand_particle_attributes: List of candidates' particleAttributes parameters (optional)
        :return: 2-tuple with list of fitness function values (minus mean packing fraction) for respective phenotype
                 candidates and list of return codes of RSA simulations for respective phenotype candidates. If the
                 RSA simulation for a candidate failed or was terminated, the respective candidates' value is np.NaN.
        """
        # values = np.zeros(len(pheno_candidates), dtype=np.float)
        values = np.full(shape=len(pheno_candidates), fill_value=np.NaN, dtype=np.float)
        # TODO Maybe define these attributes in constructor
        self.pool_workers_number = self.parallel_simulations_number  # Maybe it will be passed as an argument
        if not self.okeanos:
            self.remaining_pool_simulations = len(pheno_candidates)
            self.rsa_processes_stdins = {}
            # cand_sim_omp_threads = [self.omp_threads_number(sim_num, self.pool_workers_number,
            #                                                 self.parallel_threads_number)
            #                         for sim_num in range(len(pheno_candidates))]
            # It is said that multiprocessing module does not work with class instance method calls,
            # but in this case multiprocessing.pool.ThreadPool seems to work fine with the run_simulation method.
            with multiprocessing.pool.ThreadPool(processes=self.pool_workers_number) as pool:
                simulations_arguments = list(enumerate(pheno_candidates)) if cand_particle_attributes is None\
                    else list(zip(list(range(len(pheno_candidates))), pheno_candidates, cand_particle_attributes))
                return_codes = pool.starmap(self.rsa_simulation, simulations_arguments)
                # Maybe read the last packing fraction value after waiting for simulation process in
                # rsa_simulation method and check if the simulation labels are correct (which means that rsa3d program
                # successfully wrote the last line)
                # TODO Maybe add (it works - probably it doesn't need closing the pool):
                # pool.close()
                # pool.join()
            # TODO Consider returning status from Popen objects (if it is possible) and getting them from pool.map and
            #  checking, if rsa simulations finished correctly
        else:
            with MPIPoolExecutor(max_workers=self.pool_workers_number) as pool:
                candidate_nums = list(range(len(pheno_candidates)))
                none_list = [None] * len(pheno_candidates)
                simulation_nums = list(range(self.simulations_num, self.simulations_num + len(pheno_candidates)))
                simulations_arguments = list(zip(candidate_nums, pheno_candidates, none_list,
                                                 none_list, simulation_nums)) if cand_particle_attributes is None\
                    else list(zip(candidate_nums, pheno_candidates, cand_particle_attributes,
                                  none_list, simulation_nums))
                return_codes_iterator = pool.starmap(self.rsa_simulation, simulations_arguments)
            return_codes = list(return_codes_iterator)
            self.simulations_num += len(pheno_candidates)

        with open(self.output_filename, "r") as rsa_output_file:
            # TODO Maybe find more efficient or elegant solution
            # TODO Maybe iterate through lines in file in reversed order - results of the current generation should be
            #  at the end
            for line in rsa_output_file:
                evaluation_data = line.split("\t")
                evaluation_labels = evaluation_data[0].split(",")
                if int(evaluation_labels[0]) == self.CMAES.countiter:
                    candidate_num = int(evaluation_labels[1])
                    mean_packing_fraction = float(evaluation_data[1])
                    values[candidate_num] = -mean_packing_fraction
        # TODO Add checking if there exists a zero value in values list and deal with the error (in such a case
        #  record for corresponding candidate wasn't found in packing-fraction-vs-params.txt file)
        # return values.tolist()  # or list(values), because values.ndim == 1
        return list(values), return_codes

    def default_rsa_simulation(self, candidate_num: int, simulation_num: int, first_collector_num: int,
                               rsa_proc_arguments: List[str], task_submitting_time: datetime.datetime,
                               collectors_num: Optional[int] = 1, first_part_sim: Optional[bool] = False) \
            -> Tuple[int, int, int, int, int, str, int, datetime.datetime, str, np.ndarray]:
        """
        Function running collectors_num simulations using rsa_proc_arguments and waiting for rsa3d process. Meant to be
        called by run_simulations_on_okeanos method.

        :param candidate_num: Identification number of the candidate
        :param simulation_num: Identification number of the simulation
        :param first_collector_num: Identification number of the first collector
        :param rsa_proc_arguments: rsa3d process' arguments, which are updated with from, collectors and appendToDat
                                   parameters
        :param task_submitting_time: datetime.datetime object representing the time of submitting the partial simulation
                                     task to the pool of workers in run_simulations_on_okeanos method. Used in a logging
                                     purpose.
        :param collectors_num: Number of collectors to generate (optional), defaults to 1
        :param first_part_sim: bool value telling if the partial simulation is the first in entire RSA simulation for
                               the phenotype candidate. Used in a logging purpose.
        :return: DefaultRSASimulationResult namedtuple object containing result of the partial simulation
        """

        sim_start_time = datetime.datetime.now()
        return_code: int = 0
        particles_numbers: list = []
        pid: int = -1
        node_message: str = ""
        time: str = ""
        if self.okeanos_parallel:
            try:
                node_id = subprocess.check_output(["hostname"]).decode().strip()
            except subprocess.CalledProcessError as exception:
                self.logger.warning(msg="subprocess.CalledProcessError raised when checking host name:"
                                        " command: {}, return code: {}, output: \"{}\".\n{}\n"
                                        "Node ID will not be logged.".format(exception.cmd,
                                                                             -exception.returncode,
                                                                             exception.output.decode(),
                                                                             traceback.format_exc(limit=6).strip()))
                node_id = "not checked"
            except Exception as exception:
                self.logger.warning(msg="Exception raised when checking host name; {}: {}\n"
                                        "{}".format(type(exception).__name__, exception,
                                                    traceback.format_exc(limit=6).strip()))
                node_id = "not checked"
            node_message = "NID: {}, ".format(node_id)
        try:
            if first_part_sim:  # first_collector_num == 0
                self.logger.info(msg="\nRSA simulation start: generation number {}, candidate number {},"
                                     " simulation number {}{}\n".format(self.CMAES.countiter,
                                                                        candidate_num,
                                                                        simulation_num,
                                                                        ", " + node_message[:-2]))
            # Prepare RSA process arguments
            rsa_proc_arguments = rsa_proc_arguments[:]  # Maybe it is not needed
            rsa_proc_arguments.extend(["-from=" + str(first_collector_num), "-collectors=" + str(collectors_num),
                                       "-appendToDat=true"])
            # Create a file for saving the output of rsa3d program
            simulation_output_dir = self.rsa_output_dir + "/" + str(self.CMAES.countiter) + "_" + str(candidate_num) \
                + "_" + str(simulation_num)
            collectors_filename_info = "collector"
            if collectors_num == 1:
                collectors_filename_info += "-" + str(first_collector_num)
            else:
                collectors_filename_info += "s-{}-{}".format(first_collector_num, first_collector_num + collectors_num)
            rsa_output_filename = simulation_output_dir + "/rsa-{}-output.txt".format(collectors_filename_info)
            with open(rsa_output_filename, "w+") as rsa_output_file:
                # Open a process with simulation
                with subprocess.Popen(rsa_proc_arguments,
                                      # stdin=subprocess.PIPE,
                                      stdout=rsa_output_file,
                                      stderr=rsa_output_file,
                                      cwd=simulation_output_dir) as rsa_process:
                    pid = rsa_process.pid
                    self.logger.info(msg="RSA part. sim. start: gen. no. {}, cand. no. {}, sim. no. {},"
                                         " first col. no.: {}, collectors: {}."
                                         " {}PID: {}, task beg. delay: {}".format(self.CMAES.countiter,
                                                                                  candidate_num,
                                                                                  simulation_num,
                                                                                  first_collector_num,
                                                                                  collectors_num,
                                                                                  node_message,
                                                                                  pid,
                                                                                  sim_start_time
                                                                                  - task_submitting_time))
                    self.logger.debug(msg="RSA process arguments: {}".format(" ".join(rsa_proc_arguments)))
                    # For debugging
                    # self.logger.debug(msg="RSA simulation process call: {}".format(" ".join(rsa_proc_arguments)))
                    return_code = rsa_process.wait()
            sim_end_time = datetime.datetime.now()
            time = str(sim_end_time - sim_start_time)
            if return_code != 0:
                self.logger.warning(msg="RSA partial simulation for generation no. {}, candidate no. {},"
                                        " simulation no. {}, first collector number: {}, collectors number: {}"
                                        " returned code: {}. {}PID: {}, time: {}.\nCollectors data"
                                        " will be ignored.".format(self.CMAES.countiter,
                                                                   candidate_num,
                                                                   simulation_num,
                                                                   first_collector_num,
                                                                   collectors_num,
                                                                   return_code,
                                                                   node_message,
                                                                   pid,
                                                                   time))
            # Read RSA data file and get collectors' particle numbers
            rsa_data_file_glob = glob.glob(simulation_output_dir + "/*.dat")
            if len(rsa_data_file_glob) == 0:
                raise FileNotFoundError("RSA data file not found in the simulation directory")
            with FileReadBackwards(rsa_data_file_glob[0]) as rsa_data_file:
                # Get lines one by one starting from the last line up
                for line in rsa_data_file:
                    collector_data = line.split("\t")
                    collector_num = int(collector_data[0])
                    if first_collector_num <= collector_num < first_collector_num + collectors_num:
                        particles_numbers.insert(0, int(collector_data[1]))
                        if collector_num == first_collector_num:
                            break
            self.logger.debug(msg="RSA partial simulation end: generation no. {}, candidate no. {}, simulation no. {},"
                                  " first collector number: {}, collectors number: {},"
                                  " {}PID: {}. Time: {}, return code: {}\n"
                                  "Read collectors' particles numbers: {}".format(self.CMAES.countiter,
                                                                                  candidate_num,
                                                                                  simulation_num,
                                                                                  first_collector_num,
                                                                                  collectors_num,
                                                                                  node_message,
                                                                                  pid,
                                                                                  time,
                                                                                  return_code,
                                                                                  ", ".join(map(str,
                                                                                                particles_numbers))))
        except Exception as exception:
            self.logger.warning(msg="Exception raised in default_rsa_simulation method for generation no. {},"
                                    " candidate no. {}, simulation no. {}, first collector number: {},"
                                    " collectors number: {}{}; {}: {}\n{}\nCollectors data"
                                    " will be ignored.".format(self.CMAES.countiter, candidate_num, simulation_num,
                                                               first_collector_num, collectors_num, node_message,
                                                               type(exception).__name__, exception,
                                                               traceback.format_exc(limit=6).strip()))
            return_code = -1 if return_code == 0 else return_code
        result = DefaultRSASimulationResult(candidate_num=candidate_num,
                                            simulation_num=simulation_num,
                                            first_collector_num=first_collector_num,
                                            collectors_num=collectors_num,
                                            return_code=return_code,
                                            node_message=node_message,
                                            pid=pid,
                                            start_time=sim_start_time,
                                            time=time,
                                            particles_numbers=np.array(particles_numbers))
        # return candidate_num, simulation_num, first_collector_num, collectors_num, return_code, node_message, pid, \
        #     sim_start_time, time, np.array(particles_numbers)
        return result

    def run_simulations_on_okeanos(self, pheno_candidates: List[np.ndarray],
                                   cand_particle_attributes: Optional[List[str]] = None,
                                   candidates_numbers: Optional[List[int]] = None) \
            -> Tuple[List[float], List[int]]:
        """
        Method running rsa simulations for phenotype candidates given in pheno_candidates parameter in parallel using
        self.nodes_number - 1 Okeanos nodes as workers. Each node evaluates self.collectors_per_task collectors at once
        during a partial simulation using self.default_rsa_simulation method. Worker nodes number can be bigger than,
        equal to or smaller than the candidates number. Each simulation ends if mean packing fraction standard deviation
        is equal to or smaller than self.accuracy, unless collectors number is smaller than self.min_collectors_number.
        The method uses mpi4py.futures.MPIPoolExecutor to submit tasks (partial simulations) to processes on worker
        nodes. After submitting initial tasks, results are managed and subsequent tasks are submitted using manage_tasks
        function as concurrent.futures.Future objects' callbacks. Since mpi4py.futures.MPIPoolExecutor is used, to
        guarantee that worker processes will be spawned on different nodes, SLURM job has to specify number of nodes and
        set ntasks-per-node option to 1. Then optrsa module has to be run using srun python -m mpi4py.futures -m optrsa
        (optrsa arguments).

        :param pheno_candidates: List of NumPy ndarrays containing phenotype candidates
        :param cand_particle_attributes: List of candidates' particleAttributes parameters (optional). If not given,
                                         they will be calculated in parallel. It has to be of the same length as
                                         pheno_candidates.
        :param candidates_numbers: List of candidates numbers (optional). If not given, they will be set to the indices
                                   of the pheno_candidates list. This argument can be used to specify candidates'
                                   numbers when repeating RSA simulations for several candidates in generation, it has
                                   to be of the same length as pheno_candidates.
        :return: 2-tuple with list of fitness function values (minus mean packing fraction) for phenotype candidates and
                 list of return codes of RSA simulations for phenotype candidates. If RSA simulation for a candidate
                 failed or was terminated, the candidate's value is np.NaN.
        """

        rsa_proc_arguments = self.rsa_proc_arguments[:]  # Copy the values of the template arguments
        simulations_num = len(pheno_candidates)
        if cand_particle_attributes is None:
            # Compute candidates' particleAttributes parameters in parallel
            with multiprocessing.pool.ThreadPool(processes=min(simulations_num, os.cpu_count())) as pool:
                cand_particle_attributes = pool.map(self.arg_to_particle_attributes, pheno_candidates)
        if candidates_numbers is None:
            candidates_numbers = list(range(simulations_num))
        simulations_numbers = list(range(self.simulations_num, self.simulations_num + simulations_num))
        # TODO Maybe move it to the set_rsa_proc_arguments method
        rsa_proc_arguments.append("-ompThreads=" + str(self.parallel_threads_number))
        # For each simulation create a subdirectory for output of rsa3d program and rsa3d input file
        for candidate_num, simulation_num, particle_attributes in zip(candidates_numbers, simulations_numbers,
                                                                      cand_particle_attributes):
            simulation_output_dir = self.rsa_output_dir + "/" + str(self.CMAES.countiter) + "_" + str(candidate_num) \
                + "_" + str(simulation_num)
            if not os.path.exists(simulation_output_dir):
                os.makedirs(simulation_output_dir)
            # Create rsa3d input file containing simulation-specific parameters in the simulation output directory
            with open(simulation_output_dir + "/rsa-simulation-input.txt", "w") as rsa_input_file:
                rsa_input_file.write("ompThreads = {}\n".format(str(self.parallel_threads_number)))
                rsa_input_file.write("particleAttributes = {}\n".format(particle_attributes))

        values = np.full(shape=simulations_num, fill_value=np.NaN, dtype=np.float)
        return_codes = np.full(shape=simulations_num, fill_value=-1, dtype=np.int)
        self.pool_workers_number = self.nodes_number - 1  # Maybe it will be passed as an argument

        self.logger.info(msg="Generation no. {}, running {} simulations"
                             " using {} nodes".format(self.CMAES.countiter, simulations_num, self.pool_workers_number))
        for candidate_num, simulation_num, arg, particle_attributes in zip(candidates_numbers, simulations_numbers,
                                                                           pheno_candidates, cand_particle_attributes):
            self.logger.info(msg="Candidate no. {}, simulation no. {}".format(candidate_num, simulation_num))
            self.logger.info(msg="Argument: {}".format(arg))  # pprint.pformat(arg)
            self.logger.info(msg="particleAttributes: {}".format(particle_attributes))

        def simulation_nodes_number(simulation_number: int, parallel_simulations_number: int) -> int:
            """
            Method calculating number of nodes to assign to RSA simulation

            :param simulation_number: Number (index) of simulation from range [0, parallel_simulations_number - 1]
            :param parallel_simulations_number: Number of parallel running simulations
            :return: Number of nodes to assign to RSA simulation
            """

            sim_nodes_number = self.pool_workers_number // parallel_simulations_number
            if simulation_number < self.pool_workers_number - parallel_simulations_number * sim_nodes_number:
                sim_nodes_number += 1
            return sim_nodes_number

        # TODO Maybe add adjusting collectors_per_task value
        # simulations_nodes_numbers = np.ndarray([simulation_nodes_number(sim_num, simulations_num)
        #                                         for sim_num in range(simulations_num)])
        # active_simulations_indices = np.arange(simulations_num)
        # ending_sim_lock = multiprocessing.Lock()
        ending_sim_lock = threading.Lock()

        class OkeanosSimulation:

            collectors_ids: np.ndarray  # list of used collectors' numbers
            particles_nums: np.ndarray  # list of the corresponding particles' numbers
            packing_frac: float  # mean packing fraction
            standard_dev: float  # mean packing fraction standard deviation
            next_collector: int  # the collector index for the next task to be submitted (first free index not
            # corresponding to a submitted task)
            # data_lock: multiprocessing.Lock
            data_lock: threading.Lock

            start_time: Union[None, datetime.datetime]
            pending_part_sims: int
            nodes_number: int
            active: bool
            tasks: list

            def __init__(self):
                self.collectors_ids = np.empty(shape=0, dtype=np.int)
                self.particles_nums = np.empty(shape=0, dtype=np.int)
                self.packing_frac = 0
                self.standard_dev = 0
                self.next_collector = 0
                # self.data_lock = multiprocessing.Lock()
                self.data_lock = threading.Lock()

                self.start_time = None
                self.pending_part_sims = 0
                self.nodes_number = 0
                self.active = True
                self.tasks = []

        sims_data = [OkeanosSimulation() for _ in range(simulations_num)]
        packing_area = float(self.rsa_parameters["surfaceVolume"]) if self.input_given \
            else float(self.all_rsa_parameters["surfaceVolume"])
        # self.accuracy
        pending_simulations = simulations_num
        # evaluation_finished = multiprocessing.Event()
        # After the end of evaluation program was still waiting when using multiprocessing.Event()
        # TODO Maybe try using threading.Event() instead, maybe try using threading (or multiprocessing) Condition
        normal_evaluation = True

        # Solution inspired by https://stackoverflow.com/questions/51879070/python-executor-spawn-tasks-from-done
        # -callback-recursively-submit-tasks
        with MPIPoolExecutor(max_workers=self.pool_workers_number) as pool:

            def manage_tasks(future: Future) -> None:
                try:
                    nonlocal pending_simulations, normal_evaluation  # , sims_data
                    if not normal_evaluation:
                        return
                    if future.cancelled():
                        for sim in sims_data:
                            if future in sim.tasks:
                                sim.pending_part_sims -= 1
                                sim.tasks.remove(future)
                                break
                        # if pending_simulations == 0:  # Maybe it is not needed
                        #     evaluation_finished.set()
                        return
                    # try:
                    #     (candidate_num, first_collector_num, collectors_num, return_code,
                    #      particles_numbers) = future.result()
                    # except Exception as exception:
                    #     # TODO Maybe change it in order to avoid infinite recursion
                    #     self.logger.warning(msg="Exception raised while getting the future's result in manage_tasks"
                    #                             " function {}: {}\n{}\nThe task will be"
                    #                             " submitted again.".format(type(exception).__name__, exception,
                    #                                                        traceback.format_exc(limit=6).strip()))
                    #     pool.submit(self.default_rsa_simulation, )  # We probably don't know the task's parameters
                    res: DefaultRSASimulationResult = future.result()  # During a normal operation,
                    # default_rsa_simulation method should catch every exception
                    # (candidate_num, simulation_num, first_collector_num, collectors_num, return_code, node_message,
                    #  pid, start_time, time, particles_numbers) = future.result()
                    # TODO Maybe store future objects in sim.tasks together with the call arguments
                    #  - default_rsa_simulation method would not have to return them. Then find simulation owning the
                    #  future:
                    #  for sim in sims_data:
                    #      if future in sim.tasks:
                    #          sim.tasks.index(future)
                    #          break
                    simulation_index = res.simulation_num - self.simulations_num
                    sim = sims_data[simulation_index]
                    # if sim.start_time is None:  # res.first_collector_num == 0
                    #     sim.start_time = res.start_time
                    # One does not need to return res.start_time now
                    sim.pending_part_sims -= 1
                    sim.tasks.remove(future)
                    data_lock_message = ""
                    if not sim.data_lock.acquire(blocking=False):  # if not sim.data_lock.acquire(block=False):
                        self.logger.info(msg="Waiting for the data lock. {}".format(res.node_message[:-2]))
                        lock_pending_start = datetime.datetime.now()
                        sim.data_lock.acquire()
                        lock_pending_end = datetime.datetime.now()
                        data_lock_message = "Waiting for lock acquiring time: {}. ".format(lock_pending_end
                                                                                           - lock_pending_start)
                    # time.sleep(2)  # Testing the data lock
                    # Apparently locks aren't necessary, because callbacks are most probably executed sequentially
                    # Locks would be necessary if the future's callback function would asynchronously spawn another
                    # thread that would call manage_tasks function
                    if not sim.active:
                        # TODO Maybe don't ignore this results
                        self.logger.info(msg="RSA part. sim. end: gen. no. {}, cand. no. {}, sim. no. {},"
                                             " first col. no.: {}, collectors: {}, {}PID: {}, time: {}, ret. code: {}."
                                             " {}Pend. part. sims.: {}/{}. Simulation is not active,"
                                             " result will be ignored".format(self.CMAES.countiter,
                                                                              res.candidate_num,
                                                                              res.simulation_num,
                                                                              res.first_collector_num,
                                                                              res.collectors_num,
                                                                              res.node_message,
                                                                              res.pid,
                                                                              res.time,
                                                                              res.return_code,
                                                                              data_lock_message,
                                                                              sim.pending_part_sims,
                                                                              sim.nodes_number))
                        sim.data_lock.release()
                        # if pending_simulations == 0:
                        #     evaluation_finished.set()
                        return
                    if res.return_code != 0:
                        # TODO Maybe change it in order to avoid infinite recursion
                        # Warning was logged in the default_rsa_simulation method
                        # Submit a new task
                        sim.data_lock.release()
                        submit_task(simulation_index)
                        return
                    # Calculate new mean packing fraction and standard deviation and submit a new task, if the standard
                    # deviation is bigger than the accuracy, or end the simulation if it is smaller or equal to accuracy
                    # TODO If manage_tasks will be run in a separate thread, check if locking works and sims_data cannot
                    #  be changed during calculations
                    prev_collectors_number = sim.particles_nums.size
                    cur_collectors_number = prev_collectors_number + res.particles_numbers.size
                    sim.packing_frac = (sim.packing_frac * prev_collectors_number + np.sum(res.particles_numbers)
                                        / packing_area) / cur_collectors_number
                    collectors_indices = np.arange(res.first_collector_num,
                                                   res.first_collector_num + res.collectors_num)
                    sim.collectors_ids = np.concatenate((sim.collectors_ids, collectors_indices))
                    sim.particles_nums = np.concatenate((sim.particles_nums, res.particles_numbers))
                    # TODO Implement calculating running mean standard deviation
                    if cur_collectors_number > 1:
                        sim.standard_dev = np.sqrt(np.sum(np.power(sim.particles_nums / packing_area - sim.packing_frac,
                                                                   2))
                                                   / (cur_collectors_number - 1) / cur_collectors_number)
                    if sim.standard_dev > self.accuracy or cur_collectors_number < self.min_collectors_number:
                        self.logger.info(msg="RSA part. sim. end: gen. no. {}, cand. no. {}, sim. no. {},"
                                             " first col. no.: {}, collectors: {}. {}Current pack. frac.: {},"
                                             " cur. std. dev.: {}, pend. part. sims.: {}/{}. {}PID: {}, time: {},"
                                             " ret. code: {}".format(self.CMAES.countiter,
                                                                     res.candidate_num,
                                                                     res.simulation_num,
                                                                     res.first_collector_num,
                                                                     res.collectors_num,
                                                                     data_lock_message,
                                                                     sim.packing_frac,
                                                                     sim.standard_dev,
                                                                     sim.pending_part_sims,
                                                                     sim.nodes_number,
                                                                     res.node_message,
                                                                     res.pid,
                                                                     res.time,
                                                                     res.return_code))
                        sim.data_lock.release()
                        # Submit a new task
                        submit_task(simulation_index)
                        return
                    else:
                        # End of the RSA simulation for the candidate
                        # TODO Maybe test storing candidates' future objects, maybe cancel pending tasks belonging to
                        #  an inactive simulation, maybe after the initial submits (which number is equal to the nodes
                        #  number) submit some additional tasks in order to feed pool of workers without delays
                        # TODO If manage_tasks will be run in a separate thread, check if locking works and
                        #  pending_simulations and active attributes are not changed during calculations
                        ending_sim_lock_message = ""
                        if not ending_sim_lock.acquire(blocking=False):  # if not ending_sim_lock.acquire(block=False):
                            self.logger.info(msg="Waiting for the ending simulation lock."
                                                 " {}".format(res.node_message[:-2]))
                            lock_pending_start = datetime.datetime.now()
                            ending_sim_lock.acquire()
                            lock_pending_end = datetime.datetime.now()
                            ending_sim_lock_message = "waiting for lock" \
                                                      " acquiring time: {}, ".format(lock_pending_end
                                                                                     - lock_pending_start)
                        # time.sleep(3)  # Testing the ending simulation lock
                        sim.active = False
                        sim.data_lock.release()
                        pending_simulations -= 1
                        values[simulation_index] = -sim.packing_frac
                        return_codes[simulation_index] = 0
                        # Adjust numbers of nodes assigned to the remaining simulations
                        sim_num = 0
                        nodes_message = "\nRemaining parallel simulations: {}. Increasing numbers of nodes assigned" \
                                        " to the simulations.\n" \
                                        "simulation number: nodes number".format(pending_simulations)
                        for sim_index, simulation in enumerate(sims_data):
                            if simulation.active:
                                prev_nodes_number = simulation.nodes_number
                                simulation.nodes_number = simulation_nodes_number(sim_num, pending_simulations)
                                for _ in range(simulation.nodes_number - prev_nodes_number):
                                    submit_task(sim_index)
                                nodes_message += "\n{}: {}".format(simulations_numbers[sim_index],
                                                                   simulation.nodes_number)
                                sim_num += 1
                        self.logger.info(msg="RSA part. sim. end: gen. no. {}, cand. no. {}, sim. no. {},"
                                             " first col. no.: {}, collectors: {}. {}Current pack. frac.: {},"
                                             " cur. std. dev.: {}, pend. part. sims.: {}/{}."
                                             " {}PID: {}, time: {}, ret. code: {}\n"
                                             "\nRSA simulation end. Time: {}, collectors: {},"
                                             " {}return code: 0\n{}\n".format(self.CMAES.countiter,
                                                                              res.candidate_num,
                                                                              res.simulation_num,
                                                                              res.first_collector_num,
                                                                              res.collectors_num,
                                                                              data_lock_message,
                                                                              sim.packing_frac,
                                                                              sim.standard_dev,
                                                                              sim.pending_part_sims,
                                                                              sim.nodes_number,
                                                                              res.node_message,
                                                                              res.pid,
                                                                              res.time,
                                                                              res.return_code,
                                                                              datetime.datetime.now() - sim.start_time,
                                                                              sim.particles_nums.size,
                                                                              ending_sim_lock_message,
                                                                              nodes_message))
                        ending_sim_lock.release()
                        # if pending_simulations == 0:
                        #     evaluation_finished.set()
                        return
                except Exception as exception:
                    # Check the future result and the simulation data
                    future_result_message = "not available"
                    simulation_data_message = "not found"
                    if "res" in locals():
                        future_result_message = str(res)
                        if isinstance(res.simulation_num, int):
                            simulation_index = res.simulation_num - self.simulations_num
                            sim = sims_data[simulation_index]
                            simulation_data_message = pprint.pformat(vars(sim))
                    # Release any lock that is acquired
                    for sim in sims_data:
                        if sim.data_lock.locked():
                            sim.data_lock.release()
                    if ending_sim_lock.locked():
                        ending_sim_lock.release()
                    self.logger.warning(msg="Exception raised in manage_tasks function; {}: {}\n"
                                            "{}\nFuture result: {}\nSimulation data: {}\nSimulations that have not"
                                            " ended yet will be repeated".format(type(exception).__name__, exception,
                                                                                 traceback.format_exc(limit=6).strip(),
                                                                                 future_result_message,
                                                                                 simulation_data_message))
                    # evaluation_finished.set()
                    normal_evaluation = False  # TODO If more tasks than workers will be submitted, cancel other tasks
                    return

            def submit_task(simulation_index: int):
                sim = sims_data[simulation_index]
                first_part_sim = sim.start_time is None
                if first_part_sim:
                    sim.start_time = datetime.datetime.now()
                future = pool.submit(self.default_rsa_simulation,
                                     candidates_numbers[simulation_index],
                                     simulations_numbers[simulation_index],
                                     sim.next_collector,
                                     rsa_proc_arguments + ["-particleAttributes="
                                                           + cand_particle_attributes[simulation_index]],
                                     datetime.datetime.now(),
                                     self.collectors_per_task,
                                     first_part_sim)
                future.add_done_callback(manage_tasks)
                sim.next_collector += self.collectors_per_task
                sim.pending_part_sims += 1
                sim.tasks.append(future)

            # Submit initial tasks
            nodes_message = "Initial numbers of nodes assigned to the simulations:\n" \
                            "simulation number: nodes number".format(pending_simulations)
            for simulation_index, sim in enumerate(sims_data):
                sim.nodes_number = simulation_nodes_number(simulation_index, pending_simulations)
                for _ in range(sim.nodes_number):
                    submit_task(simulation_index)
                nodes_message += "\n{}: {}".format(simulations_numbers[simulation_index], sim.nodes_number)
            self.logger.info(msg=nodes_message + "\n")

            # evaluation_finished.wait()
            while pending_simulations > 0 and normal_evaluation:
                time.sleep(1)
            # pool.shutdown()

        if pending_simulations != 0:
            # TODO Maybe set return codes other than -1
            active_simulations = []
            for simulation_index, sim in enumerate(sims_data):
                if sim.active:
                    active_simulations.append(candidates_numbers[simulation_index])
            self.logger.warning(msg="RSA simulations for candidates no. {}"
                                    " will be repeated".format(", ".join(map(str, active_simulations))))

        self.simulations_num += len(pheno_candidates)

        # Print evaluation results to the output file
        with open(self.output_filename, "a") as output_file:
            for arg, candidate_num, simulation_num, particle_attributes, sim in zip(pheno_candidates,
                                                                                    candidates_numbers,
                                                                                    simulations_numbers,
                                                                                    cand_particle_attributes,
                                                                                    sims_data):
                if not sim.active:
                    output_file.write("{},{},{},{}"
                                      "\t{}\t{}\t{}\t{}\n".format(self.CMAES.countiter,
                                                                  candidate_num,
                                                                  simulation_num,
                                                                  " ".join(map(str, arg)),
                                                                  sim.packing_frac,
                                                                  sim.standard_dev,
                                                                  self.mode_rsa_parameters["particleType"],
                                                                  particle_attributes))

        # For each simulation, if there are unused collectors, print their indices to a file
        for simulation_index, sim in enumerate(sims_data):
            unused_collectors_ids = np.setdiff1d(np.arange(sim.next_collector), sim.collectors_ids,
                                                 assume_unique=True)
            # Alternative solution:
            # unused_collectors_ids = [id for id in range(sim.next_collector) if id not in sim.collectors_ids]
            # More efficient alternative solution that could be used if sim.collectors_ids were sorted:
            # unused_collectors_ids = []
            # prev_collector_id = -1
            # for collector_id in np.append(sim.collectors_ids, sim.next_collector):
            #     if collector_id - prev_collector_id > 1:
            #         unused_collectors_ids.extend(range(prev_collector_id + 1, collector_id))
            #     prev_collector_id = collector_id
            if unused_collectors_ids.size > 0:
                # TODO Maybe log a warning
                simulation_output_dir = self.rsa_output_dir + "/" + str(self.CMAES.countiter) \
                                        + "_" + str(candidates_numbers[simulation_index]) \
                                        + "_" + str(simulations_numbers[simulation_index])
                with open(simulation_output_dir + "/unused_collectors.txt", "w") as unused_collectors_file:
                    unused_collectors_file.writelines("{}\n".format(id) for id in unused_collectors_ids)

        return list(values), list(return_codes)

    def log_generation_data(self) -> None:
        func_data = pd.DataFrame(columns=["arg", "partattrs", "pfrac", "pfracstddev"])
        with open(self.output_filename) as output_file:
            # TODO Maybe find more efficient or elegant solution
            # TODO Maybe iterate through lines in file in reversed order - results of the current generation should be
            #  at the end
            for line in output_file:
                # Warning: when reading text files this way, each line contains "\n" character at the end - it can be
                # seen when calling print(line.__repr__()). line.rstrip("\n") should be used. Maybe use csv.reader or
                # csv.DictReader to read data files instead.
                evaluation_data = line.rstrip("\n").split("\t")
                evaluation_labels = evaluation_data[0].split(",")
                if int(evaluation_labels[0]) == self.CMAES.countiter:
                    candidate_num = int(evaluation_labels[1])
                    # If multiple lines in packing-fraction-vs-params.txt file correspond to the same candidate, the
                    # values from the last such line will be used
                    # func_data[candidate_num] = [evaluation_data[4],
                    # float(evaluation_data[1]), float(evaluation_data[2])]
                    func_data.loc[candidate_num] = [evaluation_labels[3],
                                                    evaluation_data[4],
                                                    float(evaluation_data[1]),
                                                    float(evaluation_data[2])]
        func_data.sort_values(by="pfrac", ascending=False, inplace=True)
        best_cand = func_data.iloc[0]
        median_cand = func_data.iloc[func_data.shape[0] // 2]
        worst_cand = func_data.iloc[-1]
        func_values_data = func_data.loc[:, ["pfrac", "pfracstddev"]]
        candidates = [val for ind, cand in func_values_data.iterrows()
                      for val in [ind, cand.at["pfrac"], cand.at["pfracstddev"]]]
        generation_data = [str(self.CMAES.countiter),
                           " ".join(map(str, self.CMAES.mean)),
                           self.arg_to_particle_attributes(self.CMAES.mean),  # " ".join(map(str, self.CMAES.mean))
                           " ".join(map(str, self.stddevs)),
                           # TODO Address the issue with save_optimization_data method, which does not have information
                           #  about covariance matrices during optimization
                           ";".join([",".join(map(str, row)) for row in self.covariance_matrix]),
                           " ".join(map(str, self.stddevs_to_particle_stddevs(self.CMAES.mean,
                                                                              self.stddevs,
                                                                              self.covariance_matrix))),
                           str(best_cand.name), best_cand.at["arg"], best_cand.at["partattrs"],
                           str(best_cand.at["pfrac"]), str(best_cand.at["pfracstddev"]),
                           str(median_cand.name), median_cand.at["arg"], median_cand.at["partattrs"],
                           str(median_cand.at["pfrac"]), str(median_cand.at["pfracstddev"]),
                           str(worst_cand.name), worst_cand.at["arg"], worst_cand.at["partattrs"],
                           str(worst_cand.at["pfrac"]), str(worst_cand.at["pfracstddev"]),
                           ",".join(map(str, candidates))]
        # Candidates' data is joined with "," rather than printed as separate fields separated by "\t", as below:
        # generation_data.extend(map(str, candidates))
        with open(self.opt_data_filename, "a") as opt_data_file:
            opt_data_file.write("\t".join(generation_data) + "\n")

    @classmethod
    def save_optimization_data(cls, signature) -> None:
        output_filename = _output_dir + "/" + signature + "/packing-fraction-vs-params.txt"
        mean_output_filename = _output_dir + "/" + signature + "/outcmaes/xmean.dat"
        stddev_output_filename = _output_dir + "/" + signature + "/outcmaes/stddev.dat"
        opt_data_filename = _output_dir + "/" + signature + "/optimization.dat"
        # Data of the first generation (no. 0) is logged in xmean.dat file, but in some of other CMA-ES output files
        # it is not logged
        generations_mean_data = np.loadtxt(fname=mean_output_filename, comments=['%', '#'])
        generations_stddev_data = np.loadtxt(fname=stddev_output_filename, comments=['%', '#'])
        # For backwards compatibility, check if arg field is present in the labels in RSA output file and if not, do not
        # save arg fields
        with open(output_filename) as output_file:
            line = output_file.readline()
            evaluation_data = line.rstrip("\n").split("\t")
            evaluation_labels = evaluation_data[0].split(",")
            old_format = len(evaluation_labels) < 4
        with open(output_filename) as output_file, open(opt_data_filename, "w+") as opt_data_file:
            # Write header line
            opt_data_file.write("\t".join(cls.optimization_data_columns) + "\n")
            gen_num = 0
            func_data = pd.DataFrame(columns=["arg", "partattrs", "pfrac", "pfracstddev"])
            # TODO Maybe find more efficient or elegant solution

            def save_generation_data() -> None:
                func_data.sort_values(by="pfrac", ascending=False, inplace=True)
                mean_arg = generations_mean_data[gen_num, 5:]
                stddevs = generations_stddev_data[gen_num, 5:]
                best_cand = func_data.iloc[0]
                median_cand = func_data.iloc[func_data.shape[0] // 2]
                worst_cand = func_data.iloc[-1]
                func_values_data = func_data.loc[:, ["pfrac", "pfracstddev"]]
                candidates = [val for ind, cand in func_values_data.iterrows()
                              for val in [ind, cand.at["pfrac"], cand.at["pfracstddev"]]]
                generation_data = [str(gen_num),
                                   " ".join(map(str, mean_arg)),
                                   cls.arg_to_particle_attributes(mean_arg),
                                   " ".join(map(str, stddevs)),
                                   # TODO Address the problem that this method does not have information about
                                   #  covariance matrices during optimization
                                   " ".join(map(str, cls.stddevs_to_particle_stddevs(mean_arg,
                                                                                     stddevs,
                                                                                     np.diag(stddevs ** 2)))),
                                   str(best_cand.name), best_cand.at["arg"], best_cand.at["partattrs"],
                                   str(best_cand.at["pfrac"]), str(best_cand.at["pfracstddev"]),
                                   str(median_cand.name), median_cand.at["arg"], median_cand.at["partattrs"],
                                   str(median_cand.at["pfrac"]), str(median_cand.at["pfracstddev"]),
                                   str(worst_cand.name), worst_cand.at["arg"], worst_cand.at["partattrs"],
                                   str(worst_cand.at["pfrac"]), str(worst_cand.at["pfracstddev"]),
                                   ",".join(map(str, candidates))]
                # Candidates' data is joined with "," rather than printed as separate fields separated by "\t",
                # as below:
                # generation_data.extend(map(str, candidates))
                opt_data_file.write("\t".join(generation_data) + "\n")

            for line in output_file:
                evaluation_data = line.rstrip("\n").split("\t")
                evaluation_labels = evaluation_data[0].split(",")
                if int(evaluation_labels[0]) > gen_num:
                    save_generation_data()
                    gen_num += 1
                    del func_data
                    func_data = pd.DataFrame(columns=["arg", "partattrs", "pfrac", "pfracstddev"])
                candidate_num = int(evaluation_labels[1])
                # If multiple lines in packing-fraction-vs-params.txt file correspond to the same candidate, the
                # values from the last such line will be used
                func_data.loc[candidate_num] = [evaluation_labels[3] if not old_format else "None",
                                                evaluation_data[4],
                                                float(evaluation_data[1]),
                                                float(evaluation_data[2])]
            # Save last generation's data
            save_generation_data()

    @classmethod
    def plot_optimization_data(cls, signature: str, config_file_name: str) -> None:
        opt_data_filename = _output_dir + "/" + signature + "/optimization.dat"
        # Prepare optimization data file if it does not exist
        if not os.path.isfile(opt_data_filename):
            cls.save_optimization_data(signature=signature)

        # Alternative solutions for loading optimization data:

        # 1) Check if solution using NumPy works:
        # with open(opt_data_filename) as opt_data_file:
        #     optimization_data = np.loadtxt(opt_data_file,
        #                                    # dtype={"names": tuple(cls.optimization_data_columns),
        #                                    #        "formats": tuple(cls.optimization_data_columns.values())},
        #                                    dtype={"names": ("generation_num", "meanpartattrs", "bestind",
        #                                                     "bestpfrac"),
        #                                           # "formats": (np.int, str, np.int, np.float)}
        #                                           "formats": ("i4", "U", "i4", "f4")},  # Debugging
        #                                    delimiter="\t",  # Debugging
        #                                    skiprows=1,  # Skip header line
        #                                    usecols=(0, 1, 2, 3)  # tuple(range(len(cls.optimization_data_columns)))
        #                                    )

        # 2) Maybe use function fread from datatable package

        # 3) Solution with standard lines reading and filling pd.DataFrame:
        # optimization_data = pd.DataFrame(opt_data_columns=list(opt_data_columns))
        # with open(opt_data_filename, "r") as opt_data_file:
        #     for line in opt_data_file:
        #         generation_data = line.split("\t")
        #         # ...
        #         # optimization_data.loc[int(generation_data[0])] = ...

        # 4) Reading with csv module works right:
        # with open(opt_data_filename, newline="") as opt_data_file:
        #     import csv
        #     # opt_data_reader = csv.reader(opt_data_file, delimiter="\t")  # Works right
        #     opt_data_reader = csv.DictReader(opt_data_file, delimiter="\t")  # Works right
        #     # If candidates' data is in separate opt_data_columns, remove "candidatesdata"
        #     # from cls.optimization_data_columns and use restkey="candidatesdata" in csv.DictReader constructor.
        #     for record in opt_data_reader:
        #         pprint.pprint(record)

        # Loading optimization data using pd.read_table
        # Alternatively pass filepath_or_buffer=opt_data_filename to pd.read_table
        with open(opt_data_filename) as opt_data_file:
            optimization_data = pd.read_table(filepath_or_buffer=opt_data_file,
                                              index_col="generationnum",
                                              dtype=cls.optimization_data_columns)
        # Debugging
        # # pd.set_option('display.max_columns', None)
        # # pd.set_option('display.max_rows', None)
        # pprint.pprint(optimization_data.dtypes)
        # pprint.pprint(optimization_data.index)
        # pprint.pprint(optimization_data)
        # pprint.pprint(optimization_data.loc[0, ["bestpfrac", "bestpfracstddev"]])
        # pprint.pprint(optimization_data.loc[0, ["bestpfrac", "bestpfracstddev"]].to_numpy())
        # print(optimization_data.loc[0, "bestpfrac"])
        # print(optimization_data.loc[0, "bestpartattrs"])
        # print(type(optimization_data.loc[0, "bestpartattrs"]))
        # # pprint.pprint(optimization_data.head(1))

        config_file_path = _output_dir + "/" + signature + "/" + config_file_name
        with open(config_file_path) as config_file:
            graph_config = yaml.full_load(config_file)
        # TODO Maybe use fig, ax = plt.subplots() and plot on axes
        # fig, ax = plt.subplots()
        # plt.rcParams["axes.autolimit_mode"] = "round_numbers"
        plt.rcParams["text.usetex"] = True
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.size"] = graph_config["font_size"]
        fig = plt.figure(num=args.signature, figsize=graph_config["graph_size"])  # figsize is given in inches
        # (10, 6.5)
        ax = plt.axes()
        # plt.title("CMA-ES optimization of RSA mean packing fraction\nof fixed-radii polydisks")
        plt.ylabel("Mean packing fraction")
        plt.xlabel("Generation number")
        # TODO Try to adjust ticks automatically
        # plt.xticks(optimization_data.index)  # Needed for small numbers of generations
        ax.tick_params(direction="in", right=True, top=True)
        # plt.plot(optimization_data.index, optimization_data["bestpfrac"], "go-", label="Best candidate's value")
        # plt.plot(optimization_data.index, optimization_data["medianpfrac"], "ro-", label="Median candidate's value")
        # plt.plot(optimization_data.index, optimization_data["worstpfrac"], "bo-", label="Worst candidate's value")
        candidates_data = [np.array(gen_cands_data.split(","), dtype=np.float).reshape(-1, 3)
                           for gen_cands_data in optimization_data["candidatesdata"]]
        for gen_num, gen_cands_data in enumerate(candidates_data):
            for cand_data in reversed(gen_cands_data[1:-1]):
                # Best and worst candidates are removed, median candidate stays, but his point is later covered
                plt.errorbar(x=optimization_data.index[gen_num], y=cand_data[1], yerr=cand_data[2],
                             fmt="k.", capsize=1.5)  # "ko"
        plt.errorbar(x=optimization_data.index, y=optimization_data["worstpfrac"],
                     yerr=optimization_data["worstpfracstddev"],
                     fmt="bo-", capsize=2, label="Worst candidate")  # barsabove=True  # Worst candidate's value
        plt.errorbar(x=optimization_data.index, y=optimization_data["medianpfrac"],
                     yerr=optimization_data["medianpfracstddev"],
                     fmt="rs-", capsize=2, label="Median candidate")  # barsabove=True
        plt.errorbar(x=optimization_data.index, y=optimization_data["bestpfrac"],
                     yerr=optimization_data["bestpfracstddev"],
                     fmt="gD-", capsize=2, label="Best candidate")  # barsabove=True
        plt.fill_between(optimization_data.index, optimization_data["worstpfrac"], optimization_data["bestpfrac"],
                         color="0.75")
        # plt.grid(axis="y")  # True, axis="y"
        plt.grid()
        handles, labels = ax.get_legend_handles_labels()
        leg = plt.legend(reversed(handles), reversed(labels))
        leg.set_draggable(True)
        # After dragging legend disappears, but reappears shifted after changing anything in the graph.
        # update="bbox" does not change this behaviour, but sets legend's position relative to figure, not axes, which
        # is bad.

        def particle_drawings_annotations(part_attrs_col: str, packing_frac_col: str = "medianpfrac",
                                          color: str = "0.5", modulo: int = 1, drawings_scale: float = 0.05,
                                          drawings_offset: Tuple[float, float] = None, vertical_alignment: float = 0.1,
                                          position: str = "point", arrows: bool = False, means: bool = False) -> None:
            """
            Annotate packing fraction data series with, draggable if necessary, particle drawings

            :param part_attrs_col: Name of the column with particle attributes in optimization_data pd.DataFrame
            :param packing_frac_col: Name of the column with mean packing fractions in optimization_data pd.DataFrame
            :param color: Color of the particle drawings
            :param modulo: Annotate points in first, last and every modulo generation
            :param drawings_scale: Length of unitary segment in drawing (drawing's scale) given in fraction
                                   of x axis width
            :param drawings_offset: Tuple specifying annotation boxes offset in fraction of axes' width and height
            :param vertical_alignment: Annotation boxes common vertical position in fraction of axes' height
            :param position: String specifying type of annotation boxes positioning. "point" - relative to annotated
                             points' positions - uses drawings_offset argument, "x" - at the same x positions as
                             annotated points and at the common y position for all annotation boxes,
                             specified by vertical_alignment argument.
            :param means: Whether to make annotations corresponding to means of the distributions - among others,
                          visualize particle attributes' standard deviations
            :return: None
            """
            # TODO Maybe scale drawings' paddings and arrows and boxes' frames relatively to graph's width and height,
            #  similarly as drawings are scaled
            #  Maybe use max(int(data_len * drawings_scale), 1) as the default modulo (data points are placed uniformly
            #  on axes, drawings' widths are approximately 1 in drawings' coordinates if in drawings' coordinates
            #  particle area is 1).
            # Factor for scaling objects drawn on matplotlib.offsetbox.DrawingArea
            scaling_factor = (ax.transAxes.transform((drawings_scale, drawings_scale))[0]
                              - ax.transAxes.transform((0, 0))[0]) \
                / fig.canvas.get_renderer().points_to_pixels(1.)
            # scaling_factor is divided by factor used in scaling of transformation applied by DrawingArea so that real
            # drawings' sizes are scaled by specified fraction of x axis.
            # See: https://matplotlib.org/_modules/matplotlib/offsetbox.html#DrawingArea.
            # Use fig.transFigure to specify fraction of figure's width instead. Try setting scaling_factor to 1 to see
            # how it behaves.
            gen_nums = list(range(0, data_len, modulo))
            if data_len - 1 not in gen_nums:
                gen_nums.pop()
                gen_nums.append(data_len - 1)
            for gen_num in gen_nums:
                part_attrs = optimization_data[part_attrs_col].at[gen_num]
                arg_col = part_attrs_col[:-9] + "arg"
                # For backwards compatibility, check if the arg_col field contains None
                arg_field = optimization_data[arg_col].at[gen_num]
                arg = np.array(arg_field.split(" "), dtype=np.float) if arg_field != "None" else None
                # In order to be able to draw all types of particles with full information about optimization process,
                # phenotype candidates have to be given here in the arg parameter to the draw_particle method. Therefore
                # phenotype candidates have to be somehow saved (currently they are written as the fourth label in the
                # RSA output file, but before they were only written to the logfile and saved in the optimization object
                # in the dictionary optimization.CMAES.archive). It is needed e.g. to draw the correct numbers of convex
                # polygon's vertices in the optimization space.
                # Get particle drawing and set properties of the arrow
                if not means:
                    drawing_area = cls.draw_particle(particle_attributes=part_attrs,
                                                     scaling_factor=scaling_factor,
                                                     color=color,
                                                     arg=arg)
                else:
                    stddevs = np.array(optimization_data["stddevs"].at[gen_num].split(" "), dtype=np.float)
                    part_stddevs = np.array(optimization_data["partstddevs"].at[gen_num].split(" "), dtype=np.float)
                    covariance_matrix = np.array([row.split(",") for row in
                                                  optimization_data["covmat"].at[gen_num].split(";")], dtype=np.float)
                    # covariance_matrix = np.diag(stddevs ** 2)
                    drawing_area = cls.draw_particle(particle_attributes=part_attrs,
                                                     scaling_factor=scaling_factor,
                                                     color=color,
                                                     arg=arg,
                                                     std_devs=stddevs,
                                                     covariance_matrix=covariance_matrix,
                                                     part_std_devs=part_stddevs)
                    arrows = False
                if arrows:
                    arrow_props = dict(arrowstyle="simple,"  # "->", "simple"
                                                  "head_length=0.2,"
                                                  "head_width=0.3,"  # 0.1, 0.5
                                                  "tail_width=0.01",  # 0.2
                                       facecolor="black",
                                       connectionstyle="arc3,"
                                                       "rad=0.3")
                else:
                    arrow_props = dict()
                # Set coordinates and positions of the annotated point and the label with drawing
                xy = (optimization_data.index[gen_num], optimization_data[packing_frac_col].at[gen_num])
                if position == "point":
                    box_coords = "data"
                    offset_x, offset_y = drawings_offset
                    xy_axes = ax.transAxes.inverted().transform(ax.transData.transform(xy))
                    # Transforming back to data coordinates and using xybox=xy_box, boxcoords="data" instead of
                    # xybox=(xy_axes[0] + offset_x, xy_axes[1] + offset_y), boxcoords="axes fraction",
                    # to assure correct shifting sensitivity (something concerning handling transforms by
                    # DraggableAnnotation with AnnotationBbox is wrongly implemented in matplotlib and using
                    # boxcoords="axes fraction" results in wrongly recalculated mouse's offsets and different
                    # sensitivities in both axes).
                    xy_box = ax.transData.inverted().transform(ax.transAxes.transform((xy_axes[0] + offset_x,
                                                                                       xy_axes[1] + offset_y)))
                    # Specifying axes' coordinates using ScaledTranslation doesn't work well:
                    # offset_trans = matplotlib.transforms.ScaledTranslation(offset_x, offset_y, ax.transAxes)\
                    #                - matplotlib.transforms.ScaledTranslation(0, 0, ax.transAxes)
                    # annotation_trans = ax.transData + offset_trans
                    # In AnnotationBbox constructor: xybox=xy, boxcoords=annotation_trans

                    # Alternative solutions (work well):
                    # 1) Specifying annotation box offset in fraction of figure's width and height:
                    # offset_trans = matplotlib.transforms.ScaledTranslation(offset_x, offset_y, fig.transFigure)
                    # annotation_trans = ax.transData + offset_trans
                    # In AnnotationBbox constructor: xybox=xy, boxcoords=annotation_trans
                    # 2) Displaying annotations at the bottom of axes (legend needs to be shifted then):
                    # In AnnotationBbox constructor: xybox=(xy[0], 0.1), boxcoords=("data", "axes fraction")
                    # 3) Specifying annotation box offset in data coordinates (needs to be adjusted depending on data):
                    # In AnnotationBbox constructor e.g.: xybox=(xy[0] + 0.2, xy[1] - 0.01), boxcoords="data"
                if position == "x":
                    box_coords = ("data", "axes fraction")
                    xy_box = (xy[0], vertical_alignment)
                    # arrow_props = dict()

                # Make annotation
                # drag_part_drawing = matplotlib.offsetbox.DraggableOffsetBox(ax, part_drawing)  # Not needed
                # Use matplotlib_shiftable_annotation.AnnotationBboxWithShifts for shiftability instead of draggability
                ab = matplotlib.offsetbox.AnnotationBbox(drawing_area,
                                                         xy=xy,
                                                         xybox=xy_box,
                                                         xycoords="data",
                                                         # boxcoords="axes fraction",
                                                         boxcoords=box_coords,
                                                         pad=0.2,  # 0.4
                                                         fontsize=12,  # 12
                                                         # bboxprops={},
                                                         arrowprops=arrow_props)
                ax.add_artist(ab)
                if position == "point":
                    # # AnnotationBbox subclasses matplotlib.text._AnnotationBase, so we can toggle draggability
                    # # using the following method:
                    ab.draggable()
                    # ab.shiftable()
                    # Maybe following is equivalent:
                    # drag_ab = matplotlib.offsetbox.DraggableAnnotation(ab)

        data_len = len(optimization_data["bestpartattrs"])
        drawings_scale = graph_config["drawings_scale"]  # 0.05
        modulo = graph_config.get("modulo")
        if modulo is None:
            modulo = max(int(data_len * drawings_scale), 1)
        particle_drawings_annotations(part_attrs_col="worstpartattrs", packing_frac_col="worstpfrac", color="b",
                                      modulo=modulo, drawings_scale=drawings_scale,
                                      position="x", vertical_alignment=graph_config["worst_position"])  # 0.1
        # particle_drawings_annotations(part_attrs_col="worstpartattrs", packing_frac_col="worstpfrac", color="b",
        #                               modulo=modulo, drawings_scale=drawings_scale, drawings_offset=(0., -0.15))
        # drawings_offset=(0.1, -0.1) (0.2, -0.3)
        # particle_drawings_annotations(part_attrs_col="medianpartattrs", packing_frac_col="medianpfrac", color="r",
        #                               modulo=modulo, drawings_scale=drawings_scale, drawings_offset=(0., -0.1))
        # # drawings_offset=(0.1, 0.) (0.2, -0.2)
        particle_drawings_annotations(part_attrs_col="medianpartattrs", packing_frac_col="medianpfrac", color="r",
                                      modulo=modulo, drawings_scale=drawings_scale,
                                      position="x", vertical_alignment=graph_config["median_position"])  # 0.2
        # vertical_alignment=0.22
        # particle_drawings_annotations(part_attrs_col="bestpartattrs", packing_frac_col="bestpfrac", color="g",
        #                               modulo=modulo, drawings_scale=drawings_scale, drawings_offset=(0., 0.1))
        # # drawings_offset = (0.1, 0.1) (0.2, -0.1)
        particle_drawings_annotations(part_attrs_col="bestpartattrs", packing_frac_col="bestpfrac", color="g",
                                      modulo=modulo, drawings_scale=drawings_scale,
                                      position="x", vertical_alignment=graph_config["best_position"])  # 0.3
        # vertical_alignment=1.08
        particle_drawings_annotations(part_attrs_col="meanpartattrs",
                                      modulo=modulo, drawings_scale=drawings_scale,
                                      position="x", vertical_alignment=graph_config["mean_position"],
                                      means=graph_config["annotate_mean_particles"])
        # 0.9  # 0.95
        # vertical_alignment=0.1
        gen_nums = list(range(0, data_len, modulo))
        if data_len - 1 not in gen_nums:
            gen_nums.append(data_len - 1)
        plt.xticks(gen_nums)
        # TODO Adjust top limit so that it corresponds to a major tick
        # ax.set_ymargin(0.1)
        plt.locator_params(axis="y", nbins=15)
        # ax.set_xlim(...)
        bottom_lim, top_lim = ax.get_ylim()
        bottom_space = graph_config["bottom_space"]  # 0.35
        top_space = graph_config["top_space"]  # 0.12
        ax.set_ylim(bottom_lim - bottom_space / (1. - top_space - bottom_space) * (top_lim - bottom_lim),
                    top_lim + top_space / (1. - top_space - bottom_space) * (top_lim - bottom_lim))
        # ax.set_ylim(bottom_lim - bottom_space / (1. - bottom_space) * (top_lim - bottom_lim), top_lim)

        # Create inset graph
        inset_ax = ax.inset_axes(graph_config["inset_origin"] + graph_config["inset_size"])
        inset_ax.set_xlim([graph_config["inset_data_x"][0] - 0.5, graph_config["inset_data_x"][1] + 0.5])
        inset_ax.set_ylim(graph_config["inset_data_y"])
        if graph_config["indicate_inset_zoom"]:
            ax.indicate_inset_zoom(inset_ax, edgecolor="k")
        inset_ax.tick_params(direction="in", right=True, top=True)
        inset_optimization_data = optimization_data[graph_config["inset_data_x"][0]:graph_config["inset_data_x"][1] + 1]
        inset_candidates_data = [np.array(gen_cands_data.split(","), dtype=np.float).reshape(-1, 3)
                                 for gen_cands_data in inset_optimization_data["candidatesdata"]]
        for gen_num, gen_cands_data in enumerate(inset_candidates_data):
            for cand_data in reversed(gen_cands_data[1:-1]):
                # Best and worst candidates are removed, median candidate stays, but his point is later covered
                inset_ax.errorbar(x=inset_optimization_data.index[gen_num], y=cand_data[1], yerr=cand_data[2],
                                  fmt="k.", capsize=1.5)  # "ko"
        inset_ax.errorbar(x=inset_optimization_data.index, y=inset_optimization_data["worstpfrac"],
                          yerr=inset_optimization_data["worstpfracstddev"],
                          fmt="bo-", capsize=2, label="Worst candidate")  # barsabove=True
        inset_ax.errorbar(x=inset_optimization_data.index, y=inset_optimization_data["medianpfrac"],
                          yerr=inset_optimization_data["medianpfracstddev"],
                          fmt="rs-", capsize=2, label="Median candidate")  # barsabove=True
        inset_ax.errorbar(x=inset_optimization_data.index, y=inset_optimization_data["bestpfrac"],
                          yerr=inset_optimization_data["bestpfracstddev"],
                          fmt="gD-", capsize=2, label="Best candidate")  # barsabove=True
        inset_ax.fill_between(inset_optimization_data.index, inset_optimization_data["worstpfrac"],
                              inset_optimization_data["bestpfrac"], color="0.75")
        # inset_ax.grid(axis="y")  # True, axis="y"
        inset_ax.locator_params(axis="x", nbins=graph_config["inset_x_ticks"])
        inset_ax.locator_params(axis="y", nbins=graph_config["inset_y_ticks"])
        inset_ax.grid()

        plt.tight_layout()
        plt.show()

    @waiting_for_graphs
    def run(self) -> None:
        """Method running optimization"""

        self.logger.info(msg="")
        if self.CMAES.countiter == 0:
            self.logger.info(msg="Start of optimization")
            self.CMAES.logger.add()
        else:
            self.logger.info(msg="Start of resumed optimization")
        if self.okeanos:
            self.logger.info(msg="")
            self.logger.info(msg="Parallel simulations number: {}".format(self.parallel_simulations_number))
            self.logger.info(msg="Parallel threads number: {}".format(self.parallel_threads_number))
        if self.okeanos_parallel:
            self.logger.info(msg="")
            self.logger.info(msg="Population size: {}".format(self.CMAES.popsize))
            self.logger.info(msg="Nodes number: {}".format(self.nodes_number))
            self.logger.info(msg="Parallel threads number: {}".format(self.parallel_threads_number))
            self.logger.info(msg="Collectors per task: {}".format(self.collectors_per_task))
            self.logger.info(msg="Target accuracy: {}".format(self.accuracy))
            self.logger.info(msg="Minimum collectors number: {}".format(self.min_collectors_number))
        while not self.CMAES.stop():
            gen_start_time = datetime.datetime.now()
            self.logger.info(msg="")
            self.logger.info(msg="Generation number {}".format(self.CMAES.countiter))
            if self.CMAES.countiter > 0:
                self.CMAES.logger.disp_header()
                self.CMAES.logger.disp([-1])
            # pheno_candidates = self.CMAES.ask()
            # TODO Check, why resampling causes problems in self.CMAES.tell method when population size
            #  is small and mirroring is used
            pheno_candidates = []
            resamplings_num = 0
            while len(pheno_candidates) < self.CMAES.popsize:
                candidate = self.CMAES.ask(number=1)[0]
                while not self.arg_in_domain(arg=candidate):
                    candidate = self.CMAES.ask(number=1)[0]
                    resamplings_num += 1
                pheno_candidates.append(candidate)
            if resamplings_num > 0:
                self.logger.info(msg="Resamplings per candidate: {}".format(str(resamplings_num / self.CMAES.popsize)))
            # TODO Maybe add a mode for plotting an image of a shape corresponding to mean candidate solution(s)
            self.logger.info(msg="Mean of the distribution:")
            self.logger.info(msg=pprint.pformat(self.CMAES.mean))
            self.logger.info(msg="Step size: {}".format(str(self.CMAES.sigma)))
            self.logger.info(msg="Standard deviations:")
            self.stddevs = self.CMAES.sigma * self.CMAES.sigma_vec.scaling * self.CMAES.sm.variances ** 0.5
            self.logger.info(msg=pprint.pformat(self.stddevs))
            self.logger.info(msg="Covariance matrix:")
            self.covariance_matrix = self.CMAES.sigma ** 2 * self.CMAES.sm.covariance_matrix
            for line in pprint.pformat(self.covariance_matrix).split("\n"):  # or .splitlines()
                self.logger.info(msg=line)
            self.logger.info(msg="Phenotype candidates:")
            for line in pprint.pformat(pheno_candidates).split("\n"):
                self.logger.info(msg=line)
            swapped_pheno_candidates = [self.swap_arg(arg) for arg in pheno_candidates]
            self.logger.info(msg="Swapped phenotype candidates:")
            for line in pprint.pformat(swapped_pheno_candidates).split("\n"):
                self.logger.info(msg=line)
            # values = self.evaluate_generation_parallel(pheno_candidates) if self.parallel\
            #     else self.evaluate_generation_serial(pheno_candidates)
            # TODO Maybe make evaluate_generation_* methods return values as np.ndarray
            if self.parallel:
                if self.particle_attributes_parallel:
                    if not self.okeanos_parallel:
                        values, return_codes = self.evaluate_generation_parallel_in_pool(pheno_candidates)
                    else:
                        values, return_codes = self.run_simulations_on_okeanos(pheno_candidates)
                else:
                    self.logger.info(msg="Computing candidates' particleAttributes parameters in series")
                    cand_particle_attributes = [self.arg_to_particle_attributes(arg) for arg in pheno_candidates]
                    self.logger.debug(msg="Candidates' particleAttributes parameters:")
                    for line in pprint.pformat(cand_particle_attributes, width=200).split("\n"):
                        self.logger.debug(msg=line)
                    if not self.okeanos_parallel:
                        values, return_codes = self.evaluate_generation_parallel_in_pool(pheno_candidates,
                                                                                         cand_particle_attributes)
                    else:
                        values, return_codes = self.run_simulations_on_okeanos(pheno_candidates,
                                                                               cand_particle_attributes)
                # TODO Implement computing it in parallel (probably using evaluate_generation_parallel_in_pool method,
                #  use also a function that for number of simulations and simulation number returns number
                #  of ompThreads)
                # TODO Maybe add an option to somehow tell the program to end optimization (e.g. kill -KILL an RSA
                #  process or send a signal to the main process)
                take_median = np.full(shape=len(pheno_candidates), fill_value=False)
                if not self.okeanos_parallel:
                    # while np.any(np.isnan(values)):
                    while np.any(np.logical_and(np.isnan(values), np.logical_not(take_median))):
                        for candidate_num, candidate, candidate_value, return_code\
                                in zip(list(range(len(pheno_candidates))), pheno_candidates, values, return_codes):
                            if np.isnan(candidate_value):
                                warning_message = "RSA simulation for candidate no. {} did not succeed." \
                                                  " Return code: {}".format(str(candidate_num),
                                                                            str(return_code))
                                signal_name = ""
                                if return_code < 0:
                                    try:
                                        system_info = subprocess.check_output(["uname", "-mrs"]).decode().strip()
                                    except Exception as exception:
                                        self.logger.warning(msg="Exception raised when checking system information;"
                                                                " {}: {}\n"
                                                                "{}".format(type(exception).__name__, exception,
                                                                            traceback.format_exc(limit=6).strip()))
                                        system_info = "not checked"
                                    okeanos_system_info = "Linux 4.12.14-150.17_5.0.86-cray_ari_s x86_64"
                                    # See https://docs.python.org/3/library/subprocess.html#subprocess.Popen.returncode
                                    if not self.okeanos and system_info != okeanos_system_info \
                                            and system_info != "not checked":
                                        if sys.platform.startswith("linux"):
                                            signal_info = subprocess.check_output("kill -l " + str(-return_code),
                                                                                  shell=True)
                                        else:
                                            signal_info = subprocess.check_output(["kill", "-l", str(-return_code)])
                                        signal_name = signal_info.decode().strip().upper()
                                    else:
                                        # "kill -l [number]" on okeanos doesn't work, see /usr/include/asm/signal.h
                                        # TODO Test it
                                        if return_code == -10:
                                            signal_name = "USR1"
                                        elif return_code == -12:
                                            signal_name = "USR2"
                                        else:
                                            signal_name = str(-return_code)
                                    warning_message += ", signal name: {}".format(signal_name)
                                # self.logger.debug(msg=signal_info)
                                # self.logger.debug(msg=signal_name)
                                self.logger.warning(msg=warning_message)
                                random_seed = self.rsa_parameters.get("seedOrigin") == "random" if self.input_given\
                                    else self.all_rsa_parameters.get("seedOrigin") == "random"
                                # if return_code_name in ["", "TERM"] or (return_code_name == "USR1"
                                #                                         and not random_seed):
                                #     self.logger.warning(msg="Resampling phenotype candidate"
                                #                             " no. {}".format(str(candidate_num)))
                                #     new_candidate = self.CMAES.ask(number=1)[0]
                                #     while not self.arg_in_domain(arg=new_candidate):
                                #         new_candidate = self.CMAES.ask(number=1)[0]
                                #     self.logger.info(msg="Resampled candidate no. {}:".format(str(candidate_num)))
                                #     self.logger.info(msg=pprint.pformat(new_candidate))
                                #     pheno_candidates[candidate_num] = new_candidate
                                #     return_codes[candidate_num] = self.rsa_simulation(candidate_num, new_candidate)
                                # elif return_code_name == "USR1" and random_seed:
                                if signal_name == "USR1" and random_seed:
                                    # To repeat RSA simulation in the same point when random seed RSA parameter is set,
                                    # kill simulation process with "kill -USR1 pid"
                                    self.logger.warning(msg="Repeating RSA simulation for phenotype candidate"
                                                            " no. {}".format(str(candidate_num)))
                                    return_codes[candidate_num] = self.rsa_simulation(
                                        candidate_num, candidate,
                                        omp_threads=self.parallel_threads_number)
                                elif signal_name == "USR2":
                                    # To set corresponding to RSA simulation phenotype candidate's value to the median
                                    # of other candidates' values, kill simulation process with "kill -USR2 pid"
                                    self.logger.warning(msg="Phenotype candidate's no. {} value will be set"
                                                            " to the median of other candidates'"
                                                            " values".format(str(candidate_num)))
                                    take_median[candidate_num] = True
                                else:
                                    # To resample phenotype candidate corresponding to RSA simulation,
                                    # kill simulation process in other way, e.g. with "kill pid"
                                    self.logger.warning(msg="Resampling phenotype candidate"
                                                            " no. {}".format(str(candidate_num)))
                                    new_candidate = self.CMAES.ask(number=1)[0]
                                    while not self.arg_in_domain(arg=new_candidate):
                                        new_candidate = self.CMAES.ask(number=1)[0]
                                    self.logger.info(msg="Resampled candidate no. {}:".format(str(candidate_num)))
                                    self.logger.info(msg=pprint.pformat(new_candidate))
                                    pheno_candidates[candidate_num] = new_candidate
                                    return_codes[candidate_num] = self.rsa_simulation(
                                        candidate_num, new_candidate,
                                        omp_threads=self.parallel_threads_number)

                        with open(self.output_filename, "r") as rsa_output_file:
                            # TODO Maybe find more efficient or elegant solution
                            # TODO Maybe iterate through lines in file in reversed order - results of the current
                            #  generation should be at the end
                            for line in rsa_output_file:
                                evaluation_data = line.split("\t")
                                evaluation_labels = evaluation_data[0].split(",")
                                if int(evaluation_labels[0]) == self.CMAES.countiter:
                                    read_candidate_num = int(evaluation_labels[1])
                                    mean_packing_fraction = float(evaluation_data[1])
                                    values[read_candidate_num] = -mean_packing_fraction
                else:
                    # TODO Test it
                    values = np.array(values)
                    return_codes = np.array(return_codes)
                    random_seed = self.rsa_parameters.get("seedOrigin") == "random" if self.input_given\
                        else self.all_rsa_parameters.get("seedOrigin") == "random"
                    while np.any(np.logical_and(np.isnan(values), np.logical_not(take_median))):
                        repeat_candidates = np.full(shape=len(pheno_candidates), fill_value=False)
                        if random_seed:
                            repeat_candidates = return_codes == -10
                            repeat_candidates_nums = [str(candidate_num) for candidate_num, repeat
                                                      in enumerate(repeat_candidates) if repeat]
                            if np.any(repeat_candidates):
                                self.logger.warning(msg="Phenotype candidates' no. {} simulations"
                                                        " will be repeated".format(", ".join(repeat_candidates_nums)))
                        median_candidates = return_codes == -12
                        if np.any(median_candidates):
                            median_candidates_nums = [candidate_num for candidate_num, median
                                                      in enumerate(median_candidates) if median]
                            self.logger.warning(msg="Phenotype candidates' no. {} values will be set"
                                                    " to the median of other candidates'"
                                                    " values".format(", ".join(map(str, median_candidates_nums))))
                            take_median[median_candidates] = True
                        # Resample candidates
                        resample_candidates = np.logical_and(np.isnan(values),
                                                             np.logical_not(take_median),
                                                             np.logical_not(repeat_candidates))
                        if np.any(resample_candidates):
                            resample_candidates_nums = [candidate_num for candidate_num, resample
                                                        in enumerate(resample_candidates) if resample]
                            resamplings_num = 0
                            for candidate_num in resample_candidates_nums:
                                new_candidate = self.CMAES.ask(number=1)[0]
                                while not self.arg_in_domain(arg=new_candidate):
                                    new_candidate = self.CMAES.ask(number=1)[0]
                                    resamplings_num += 1
                                pheno_candidates[candidate_num] = new_candidate
                            self.logger.warning(msg="Phenotype candidates no. {}"
                                                    " were resampled.".format(", ".join(map(str,
                                                                                            resample_candidates_nums))))
                            if resamplings_num > 0:
                                self.logger.info(msg="Resamplings per candidate: {}".format(
                                    resamplings_num / len(resample_candidates_nums)))
                            self.logger.info(msg="New candidates:")
                            for candidate_num in resample_candidates_nums:
                                self.logger.info(msg=pprint.pformat(pheno_candidates[candidate_num]))
                            if not self.particle_attributes_parallel:
                                self.logger.info(msg="Computing candidates' particleAttributes parameters in series")
                                self.logger.debug(msg="Candidates' particleAttributes parameters:")
                                for candidate_num in resample_candidates_nums:
                                    cand_particle_attributes[candidate_num] = self.arg_to_particle_attributes(
                                        pheno_candidates[candidate_num])
                                    self.logger.debug(msg=cand_particle_attributes[candidate_num])
                        # Evaluate rest of the candidates
                        evaluate_candidates = np.logical_and(np.isnan(values), np.logical_not(take_median))
                        candidates_to_evaluate = [candidate for candidate, evaluate
                                                  in zip(pheno_candidates, evaluate_candidates) if evaluate]
                        candidates_numbers = [candidate_num for candidate_num, evaluate
                                              in enumerate(evaluate_candidates) if evaluate]
                        self.logger.info(msg="Evaluating values for candidates"
                                             " no. {}".format(", ".join(map(str, candidates_numbers))))
                        if self.particle_attributes_parallel:
                            new_values, new_return_codes = self.run_simulations_on_okeanos(
                                candidates_to_evaluate,
                                candidates_numbers=candidates_numbers)
                        else:
                            candidates_part_attrs = [part_attrs for part_attrs, evaluate
                                                     in zip(cand_particle_attributes, evaluate_candidates) if evaluate]
                            new_values, new_return_codes = self.run_simulations_on_okeanos(
                                candidates_to_evaluate,
                                cand_particle_attributes=candidates_part_attrs,
                                candidates_numbers=candidates_numbers)
                        # Set values and return codes
                        values[evaluate_candidates] = new_values
                        return_codes[evaluate_candidates] = new_return_codes
                    # values = list(values)
                if np.any(take_median):
                    # TODO Maybe make evaluate_generation_* methods return values in np.ndarray - then it would be
                    #  easier to manipulate the values array
                    correct_values = []
                    for val in values:
                        if not np.isnan(val):
                            correct_values.append(val)
                    median_value = np.sort(correct_values)[len(correct_values) // 2]
                    self.logger.warning(msg="Phenotype candidates' no. {} values"
                                            " are set to the median of other candidates' values"
                                            " equal to {}".format(", ".join(map(str, np.nonzero(take_median)[0])),
                                                                  str(median_value)))
                    values = [value if not take_med else median_value for value, take_med in zip(values, take_median)]
            else:
                # TODO Implement checking results in serial computing case
                values = self.evaluate_generation_serial(pheno_candidates)
            # TODO Check, what happens in case when e.g. None is returned as candidate value
            # TODO Maybe add checking if rsa simulation finished with success and successfully wrote a line to
            #  packing-fraction-vs-params.txt file. If it fails, in serial computing the previous packing fraction
            #  is assigned as the current value in values array without any warning, and in parallel - wrong value
            #  from np.zeros function is treated as a packing fraction.
            self.logger.info(msg="End of generation number {}".format(self.CMAES.countiter))
            self.logger.info(msg="Candidate values:")
            self.logger.info(msg=values)
            if self.log_generations:
                # Less costly solution: read not only packing fraction, but also standard deviation from
                # packing-fraction-vs-params.txt file in evaluate_generation_* function and return both values.
                # File wouldn't be read twice then. For distribution mean, best, median and worst candidate
                # self.arg_to_particle_attributes would be called. All data read from file by self.log_generation_data
                # would be available. Standard deviations could be written to optimization-output.log file, too.
                self.log_generation_data()
            self.CMAES.tell(swapped_pheno_candidates, values)
            self.CMAES.logger.add()
            # Pickling of the object
            # TODO Add jsonpickle pickling
            self.pickle()
            gen_end_time = datetime.datetime.now()
            self.logger.info("Generation time: {}".format(str(gen_end_time - gen_start_time)))
        self.logger.info(msg="")
        self.logger.info(msg="End of optimization")
        self.logger.info(msg="")
        self.CMAES.logger.disp_header()
        self.CMAES.logger.disp([-1])
        self.logger.info(msg=pprint.pformat(self.CMAES.result))
        self.CMAES.result_pretty()
        # Pickling of the object
        self.pickle()
        termination_condition = "-".join(["{}-{}".format(key, val) for key, val in self.CMAES.result.stop.items()])\
            .replace(".", "_")
        self.pickle(name="gen-{}-term-{}".format(self.CMAES.countiter - 1, termination_condition))

        # TODO Add separate method for making graphs
        # TODO Maybe create another class for analyzing the results of optimization
        if self.show_graph:
            plot_cmaes_graph_in_background(self.CMAES.logger.name_prefix, self.signature)

        # if self.output_to_file:
        sys.stdout = self.stdout
        sys.stderr = self.stderr


class PolydiskRSACMAESOpt(RSACMAESOptimization, metaclass=abc.ABCMeta):

    # mode_rsa_parameters: dict = dict(super().mode_rsa_parameters, particleType="Polydisk")
    # TODO Check, if it is a right way of overriding class attributes (how to get parent class' attribute)
    mode_rsa_parameters: dict = dict(RSACMAESOptimization.mode_rsa_parameters,
                                     surfaceDimension="2", particleType="Polydisk")

    wolfram_polydisk_area_eval_num: int = -1

    @classmethod
    @abc.abstractmethod
    def arg_to_polydisk_attributes(cls, arg: np.ndarray) -> Tuple[str, np.ndarray]:
        """
        Function returning part of Polydisk's particleAttributes in a tuple, which first element is \"xy\" or \"rt\"
        string indicating type of coordinates and the second is a numpy ndarray with c01 c02 r0 c11 c12 r1 ... floats
        (disks' coordinates and radii)
        """
        pass

    # @classmethod
    # @abc.abstractmethod
    # def stddevs_to_polydisk_stddevs(cls, stddevs: np.ndarray) -> np.ndarray:
    #     pass

    @classmethod
    def arg_in_domain(cls, arg: np.ndarray) -> bool:
        coordinates_type, disks_params = cls.arg_to_polydisk_attributes(arg)
        # TODO Maybe do it better
        intersection_tests = {
            "xy": lambda first_disk, second_disk: np.sqrt((first_disk[0] - second_disk[0]) ** 2
                                                          + (first_disk[1] - second_disk[1]) ** 2)
                                                  <= first_disk[2] + second_disk[2],
            "rt": lambda first_disk, second_disk: np.sqrt((first_disk[0] * np.cos(first_disk[1])
                                                           - second_disk[0] * np.cos(second_disk[1])) ** 2
                                                          + (first_disk[0] * np.sin(first_disk[1])
                                                             - second_disk[0] * np.sin(second_disk[1])) ** 2)
                                                  <= first_disk[2] + second_disk[2]
        }
        if coordinates_type in intersection_tests:
            disks_intersect = intersection_tests[coordinates_type]
            disks_args = np.reshape(disks_params, (-1, 3))
            # Check, if the polydisk is connected by checking if one iteration of DFS of the corresponding
            # undirected graph visits all of the graph's vertices. Check the vertex corresponding to a free disk in
            # constrained versions of polydisks.
            disks_visited = np.full(shape=disks_args.shape[0], fill_value=False)

            # TODO Check if it is correct and sufficiently optimal
            def polydisk_dfs_visit(disk_index: int) -> None:
                current_disk_args = disks_args[disk_index]
                disks_visited[disk_index] = True
                for checked_disk_index, checked_disk_args in enumerate(disks_args):
                    if not disks_visited[checked_disk_index] and disks_intersect(checked_disk_args, current_disk_args):
                        polydisk_dfs_visit(disk_index=checked_disk_index)

            polydisk_dfs_visit(disk_index=0)
            return np.all(disks_visited)
        else:
            raise NotImplementedError("Checking if a polydisk with attributes given in {} coordinates is connected"
                                      " is not implemented yet.".format(coordinates_type))

    @classmethod
    def draw_particle(cls, particle_attributes: str, scaling_factor: float, color: str,
                      arg: Optional[np.ndarray] = None, std_devs: Optional[np.ndarray] = None,
                      covariance_matrix: Optional[np.ndarray] = None, part_std_devs: Optional[np.ndarray] = None) \
            -> matplotlib.offsetbox.DrawingArea:
        # Extract particle data
        # Scale polydisks so that they have unitary area
        part_data = np.array(particle_attributes.split(" ")[2:-1], dtype=np.float).reshape(-1, 3) \
                    / np.sqrt(np.float(particle_attributes.rpartition(" ")[2]))
        if part_std_devs is not None:
            std_devs_data = part_std_devs.reshape(-1, 3) / np.sqrt(np.float(particle_attributes.rpartition(" ")[2]))
        # Draw particle
        # Get polydisk drawing's width and height
        if part_std_devs is None:
            x_min = np.min(part_data[:, 0] - part_data[:, 2])
            x_max = np.max(part_data[:, 0] + part_data[:, 2])
            y_min = np.min(part_data[:, 1] - part_data[:, 2])
            y_max = np.max(part_data[:, 1] + part_data[:, 2])
        else:
            x_min = np.min(np.concatenate((part_data[:, 0] - part_data[:, 2], part_data[:, 0] - std_devs_data[:, 0])))
            x_max = np.max(np.concatenate((part_data[:, 0] + part_data[:, 2], part_data[:, 0] + std_devs_data[:, 0])))
            y_min = np.min(np.concatenate((part_data[:, 1] - part_data[:, 2], part_data[:, 1] - std_devs_data[:, 1])))
            y_max = np.max(np.concatenate((part_data[:, 1] + part_data[:, 2], part_data[:, 1] + std_devs_data[:, 1])))
            # TODO Take into account also radii standard deviations
        drawing_area = matplotlib.offsetbox.DrawingArea(scaling_factor * (x_max - x_min),
                                                        scaling_factor * (y_max - y_min),
                                                        scaling_factor * -x_min,
                                                        scaling_factor * -y_min)
        for disk_args in part_data:
            disk = matplotlib.patches.Circle((scaling_factor * disk_args[0], scaling_factor * disk_args[1]),
                                             scaling_factor * disk_args[2],
                                             color=color)
            # transform=matplotlib.transforms.IdentityTransform() - same as transform=None, probably different
            # than transform used when this argument is not passed.
            drawing_area.add_artist(disk)
        for disk_num, disk_args in enumerate(part_data):
            if part_std_devs is None:
                disk_label = matplotlib.text.Text(x=scaling_factor * disk_args[0], y=scaling_factor * disk_args[1],
                                                  text=str(disk_num),
                                                  horizontalalignment="center",
                                                  verticalalignment="center",
                                                  fontsize=11)
                drawing_area.add_artist(disk_label)
            else:
                disk_label = matplotlib.text.Text(x=scaling_factor * disk_args[0] + scaling_factor * disk_args[2] / 2,
                                                  y=scaling_factor * disk_args[1] + scaling_factor * disk_args[2] / 2,
                                                  text=str(disk_num),
                                                  horizontalalignment="center",
                                                  verticalalignment="center",
                                                  fontsize=9)
                drawing_area.add_artist(disk_label)

                # test_arrow = matplotlib.patches.FancyArrow(0, 0, scaling_factor * 1, scaling_factor * 1)
                # arrow_style = matplotlib.patches.ArrowStyle("simple", head_width=1.2)
                # test_arrow = matplotlib.patches.FancyArrowPatch(
                #     (0, 0),
                #     (scaling_factor * 1 / np.sqrt(np.float(particle_attributes.rpartition(" ")[2])),
                #      0),
                #     shrinkA=0,
                #     shrinkB=0)
                # drawing_area.add_artist(test_arrow)
                # test_arrow = matplotlib.patches.FancyArrowPatch(
                #     (0, 0),
                #     (scaling_factor * 1 / np.sqrt(np.float(particle_attributes.rpartition(" ")[2])) / np.sqrt(2),
                #      scaling_factor * 1 / np.sqrt(np.float(particle_attributes.rpartition(" ")[2])) / np.sqrt(2)),
                #     arrowstyle=arrow_style,
                #     shrinkA=0,
                #     shrinkB=0)
                # # test_arrow = matplotlib.patches.ConnectionPatch((0, 0), (scaling_factor * 1, scaling_factor * 1))
                # drawing_area.add_artist(test_arrow)

                # arrow_style = matplotlib.patches.ArrowStyle("simple", head_width=1.2)  # Causes a bug in matplotlib
                # arrow_style = matplotlib.patches.ArrowStyle("->", head_width=0.8)
                # Head lengths are not scaled and for small standard deviations heads are longer than arrow, so one
                # solution is to make them not visible
                # TODO Make arrows lengths correct while using arrows without heads
                arrow_style = matplotlib.patches.ArrowStyle("->", head_length=0.)
                center = (scaling_factor * disk_args[0], scaling_factor * disk_args[1])
                ticks = [(center[0] + scaling_factor * std_devs_data[disk_num][0], center[1]),
                         (center[0] - scaling_factor * std_devs_data[disk_num][0], center[1]),
                         (center[0], center[1] + scaling_factor * std_devs_data[disk_num][1]),
                         (center[0], center[1] - scaling_factor * std_devs_data[disk_num][1])]
                for tick in ticks:
                    std_dev_arrow = matplotlib.patches.FancyArrowPatch(
                        center,
                        tick,
                        arrowstyle=arrow_style,
                        shrinkA=0,
                        shrinkB=0)
                    drawing_area.add_artist(std_dev_arrow)
                # TODO Take into account also radii standard deviations
        return drawing_area

    # TODO Make it a class method?
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

    @staticmethod
    def wolframclient_polydisk_area(disks_params: np.ndarray) -> float:
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

    @classmethod
    def arg_to_particle_attributes(cls, arg: np.ndarray) -> str:
        """Function returning rsa3d program's parameter particleAttributes based on arg"""
        coordinates_type, disks_params = cls.arg_to_polydisk_attributes(arg)
        disks_num = disks_params.size // 3
        # area = cls.wolfram_polydisk_area(disks_params)
        area = cls.wolframclient_polydisk_area(disks_params)
        particle_attributes_list = [str(disks_num), coordinates_type]
        # TODO Maybe do it in a more simple way
        particle_attributes_list.extend(disks_params.astype(np.unicode).tolist())
        particle_attributes_list.append(str(area))
        return " ".join(particle_attributes_list)


class FixedRadiiXYPolydiskRSACMAESOpt(PolydiskRSACMAESOpt):
    """
    Class for performing CMA-ES optimization of packing fraction of RSA packings built of unions of disks
    with unit radius. All disks centers' coordinates are free.
    """

    default_rsa_parameters = dict(PolydiskRSACMAESOpt.default_rsa_parameters,  # super().default_rsa_parameters,
                                  **{"maxVoxels": "4000000",
                                     "requestedAngularVoxelSize": "0.3",
                                     "minDx": "0.0",
                                     "from": "0",
                                     "collectors": "5",
                                     "split": "100000",
                                     "boundaryConditions": "periodic"})

    def get_arg_signature(self) -> str:
        disks_num = self.initial_mean.size // 2
        return "disks-" + str(disks_num) + "-initstds-" + str(self.initial_stddevs)

    # TODO Check, if constructor has to be overwritten

    @classmethod
    def arg_to_particle_parameters(cls, arg: np.ndarray) -> np.ndarray:
        """
        Function returning polydisk's parameters in a numpy ndarray with c01 c02 r0 c11 c12 r1 ... floats (disks'
        coordinates and radii)
        """
        arg_with_radii = np.insert(arg, np.arange(2, arg.size + 1, 2), 1.)
        return arg_with_radii

    @classmethod
    def arg_to_polydisk_attributes(cls, arg: np.ndarray) -> Tuple[str, np.ndarray]:
        """
        Function returning part of Polydisk's particleAttributes in a tuple, which first element is \"xy\" or \"rt\"
        string indicating type of coordinates and the second is a numpy ndarray with c01 c02 r0 c11 c12 r1 ... floats
        (disks' coordinates and radii)
        """
        arg_with_radii = cls.arg_to_particle_parameters(arg)
        return "xy", arg_with_radii

    @classmethod
    def stddevs_to_particle_stddevs(cls, arg: np.ndarray, stddevs: np.ndarray, covariance_matrix: np.ndarray) \
            -> np.ndarray:
        stddevs_with_radii = np.insert(stddevs, np.arange(2, stddevs.size + 1, 2), 0.)
        return stddevs_with_radii


class ConstrFixedRadiiXYPolydiskRSACMAESOpt(PolydiskRSACMAESOpt):
    """
    Class for performing CMA-ES optimization of packing fraction of RSA packings built of unions of disks
    with unit radius. The last disk is placed at (0, 0), the last but one at (x, 0) and others are free.
    """

    default_rsa_parameters = dict(PolydiskRSACMAESOpt.default_rsa_parameters,  # super().default_rsa_parameters,
                                  **{"maxVoxels": "4000000",
                                     "requestedAngularVoxelSize": "0.3",
                                     "minDx": "0.0",
                                     "from": "0",
                                     "collectors": "5",
                                     "split": "100000",
                                     "boundaryConditions": "periodic"})

    def get_arg_signature(self) -> str:
        disks_num = (self.initial_mean.size - 1) // 2 + 2
        return "disks-" + str(disks_num) + "-initstds-" + str(self.initial_stddevs)

    # TODO Check, if constructor has to be overwritten

    @classmethod
    def arg_to_particle_parameters(cls, arg: np.ndarray) -> np.ndarray:
        """
        Function returning polydisk's parameters in a numpy ndarray with c01 c02 r0 c11 c12 r1 ... floats (disks'
        coordinates and radii)
        """
        arg_with_standard_disks_radii = np.insert(arg, np.arange(2, arg.size, 2), 1.)
        arg_with_all_disks = np.concatenate((arg_with_standard_disks_radii, np.array([0., 1., 0., 0., 1.])))
        return arg_with_all_disks

    @classmethod
    def arg_to_polydisk_attributes(cls, arg: np.ndarray) -> Tuple[str, np.ndarray]:
        """
        Function returning part of Polydisk's particleAttributes in a tuple, which first element is \"xy\" or \"rt\"
        string indicating type of coordinates and the second is a numpy ndarray with c01 c02 r0 c11 c12 r1 ... floats
        (disks' coordinates and radii)
        """
        arg_with_all_disks = cls.arg_to_particle_parameters(arg)
        return "xy", arg_with_all_disks

    @classmethod
    def stddevs_to_particle_stddevs(cls, arg: np.ndarray, stddevs: np.ndarray, covariance_matrix: np.ndarray) \
            -> np.ndarray:
        stddevs_with_standard_disks_radii = np.insert(stddevs, np.arange(2, stddevs.size, 2), 0.)
        stddevs_with_all_disks = np.concatenate((stddevs_with_standard_disks_radii, np.zeros(5, dtype=np.float)))
        return stddevs_with_all_disks


class PolygonRSACMAESOpt(RSACMAESOptimization, metaclass=abc.ABCMeta):

    mode_rsa_parameters: dict = dict(RSACMAESOptimization.mode_rsa_parameters,
                                     surfaceDimension="2", particleType="Polygon")
    # string indicating type of coordinates, e.g. "xy" or "rt"
    coordinates_type: str = None

    @classmethod
    @abc.abstractmethod
    def select_vertices(cls, points: np.ndarray) -> np.ndarray:
        """
        Function selecting polygon's vertices from points returned by the arg_to_points_coordinates method. Given
        a NumPy ndarray of shape (n, 2) with points' coordinates, it returns indices of the subsequent polygon's
        vertices in a NumPy ndarray of shape (n,) with integers
        """
        pass

    @classmethod
    @abc.abstractmethod
    def arg_to_points_coordinates(cls, arg: np.ndarray) -> np.ndarray:
        """
        Function returning coordinates of points being candidates for becoming polygon's vertices in a NumPy ndarray of
        shape (n, 2) with floats
        """
        pass

    @classmethod
    @abc.abstractmethod
    def points_coordinates_to_arg(cls, points: np.ndarray) -> np.ndarray:
        """
        Inverse of the arg_to_points_coordinates function
        """
        pass

    @classmethod
    def arg_to_particle_parameters(cls, arg: np.ndarray) -> np.ndarray:
        """
        Function returning points' (not polygon's) parameters based on arg
        """
        points_parameters = cls.arg_to_points_coordinates(arg).flatten()
        return points_parameters

    @classmethod
    def arg_to_particle_attributes(cls, arg: np.ndarray) -> str:
        """Function returning rsa3d program's parameter particleAttributes based on arg"""
        points_parameters = cls.arg_to_particle_parameters(arg)
        points_coordinates = points_parameters.reshape(-1, 2)
        vertices = points_coordinates[cls.select_vertices(points_coordinates)].flatten()
        vertices_num = vertices.size // 2
        particle_attributes_list = [str(vertices_num), cls.coordinates_type]
        particle_attributes_list.extend(vertices.astype(np.unicode).tolist())
        particle_attributes_list.append(str(vertices_num))
        particle_attributes_list.extend(np.arange(vertices_num).astype(np.unicode).tolist())
        return " ".join(particle_attributes_list)

    @classmethod
    def stddevs_to_points_stddevs(cls, arg: np.ndarray, stddevs: np.ndarray, covariance_matrix: np.ndarray) \
            -> np.ndarray:
        """
        Function returning standard deviations of points being candidates for becoming polygon's vertices in a form of
        a NumPy ndarray of shape (n, 2) with floats representing points' standard deviations. By default it calculates
        points' standard deviations by sampling.
        """
        return super().stddevs_to_particle_stddevs(arg, stddevs, covariance_matrix).reshape(-1, 2)

    @classmethod
    def stddevs_to_particle_stddevs(cls, arg: np.ndarray, stddevs: np.ndarray, covariance_matrix: np.ndarray) \
            -> np.ndarray:
        points_stddevs = cls.stddevs_to_points_stddevs(arg, stddevs, covariance_matrix)
        points_coordinates = cls.arg_to_points_coordinates(arg)
        return points_stddevs[cls.select_vertices(points_coordinates)].flatten()

    @classmethod
    def draw_particle(cls, particle_attributes: str, scaling_factor: float, color: str,
                      arg: Optional[np.ndarray] = None, std_devs: Optional[np.ndarray] = None,
                      covariance_matrix: Optional[np.ndarray] = None, part_std_devs: Optional[np.ndarray] = None) \
            -> matplotlib.offsetbox.DrawingArea:
        # TODO Implement it
        pass


class ConvexPolygonRSACMAESOpt(PolygonRSACMAESOpt, metaclass=abc.ABCMeta):

    @classmethod
    def select_vertices(cls, points: np.ndarray) -> np.ndarray:
        if cls.coordinates_type != "xy":
            # ConvexHull constructor requires Cartesian coordinates, so a conversion has to be made
            conversions = {
                "rt": lambda point: np.array([point[0] * np.cos(point[1]), point[0] * np.sin(point[1])])
            }
            if cls.coordinates_type in conversions:
                # TODO Test this conversion
                points = np.apply_along_axis(func1d=conversions[cls.coordinates_type], axis=1, arr=points)
            else:
                raise NotImplementedError("Conversion of {} coordinates into Cartesian coordinates is not implemented"
                                          "yet.".format(cls.coordinates_type))
        if np.all(points == points[0, :]):
            # Degenerate case of initializing mean of the distribution with a sequence of equal points
            # TODO Check, if this is the right thing to do
            return np.array([0])
        # TODO Maybe deal with other degenerate cases
        convex_hull = ConvexHull(points)
        return convex_hull.vertices

    @classmethod
    def swap_arg(cls, arg: np.ndarray) -> np.ndarray:
        points = cls.arg_to_points_coordinates(arg)
        if cls.coordinates_type != "xy":
            conversions = {
                "rt": lambda point: np.array([point[0] * np.cos(point[1]), point[0] * np.sin(point[1])])
            }
            if cls.coordinates_type in conversions:
                points_xy = np.apply_along_axis(func1d=conversions[cls.coordinates_type], axis=1, arr=points)
            else:
                raise NotImplementedError("Conversion of {} coordinates into Cartesian coordinates is not implemented"
                                          "yet.".format(cls.coordinates_type))
        else:
            points_xy = points
        points_num = points_xy.shape[0]
        vertices_indices = cls.select_vertices(points_xy)
        vertices_num = vertices_indices.size
        vertices = points_xy[vertices_indices]
        center = np.mean(vertices, axis=0)
        inner_points_indices = np.setdiff1d(np.arange(points_num), vertices_indices, assume_unique=True)

        transformed_inner_points_xy = np.empty(0, dtype=np.float)
        for inner_point_index in inner_points_indices:
            inner_point = points_xy[inner_point_index]
            inner_point_vector = inner_point - center
            i = 0
            while i < vertices_num - 1:
                if np.cross(vertices[i] - center, inner_point_vector) \
                        * np.cross(vertices[i + 1] - center, inner_point_vector) < 0:
                    break
                i += 1
            # TODO Handle case in which vertices[i + 1, 0] == vertices[i, 0]
            side_slope = (vertices[i + 1, 1] - vertices[i, 1]) / (vertices[i + 1, 0] - vertices[i, 0])
            transf_line_slope = inner_point_vector[1] / inner_point_vector[0]
            transf_inner_point_x = (transf_line_slope * center[0] - center[1]
                                    - side_slope * vertices[i, 0] + vertices[i, 1]) / (transf_line_slope - side_slope)
            transf_inner_point_y = transf_line_slope * (transf_inner_point_x - center[0]) + center[1]
            transformed_inner_point_xy = np.array([transf_inner_point_x, transf_inner_point_y])
            transformed_inner_points_xy = np.append(transformed_inner_points_xy, transformed_inner_point_xy)

        if cls.coordinates_type != "xy":
            conversions = {
                "rt": lambda point: np.array([np.sqrt(point[0] ** 2 + point[1] ** 2),
                                              np.arctan(point[1] / point[0]) if point[0] != 0
                                              else np.pi / 2 if point[1] > 0
                                              else 3 * np.pi / 2 if point[1] < 0 else 0])
            }
            if cls.coordinates_type in conversions:
                transformed_inner_points = np.apply_along_axis(func1d=conversions[cls.coordinates_type],
                                                               axis=1,
                                                               arr=transformed_inner_points_xy)
            else:
                raise NotImplementedError("Conversion of Cartesian coordinates into {} coordinates is not implemented"
                                          "yet.".format(cls.coordinates_type))
        else:
            transformed_inner_points = transformed_inner_points_xy

        points[inner_points_indices] = transformed_inner_points
        swapped_arg = cls.points_coordinates_to_arg(points)
        return swapped_arg


class StarShapedPolygonRSACMAESOpt(PolygonRSACMAESOpt, metaclass=abc.ABCMeta):

    @classmethod
    def select_vertices(cls, points: np.ndarray) -> np.ndarray:
        points = np.copy(points)  # NumPy arrays are passed by reference, so this prevents modifying passed object
        if cls.coordinates_type != "xy":
            # Polygonization algorithm converts coordinates to radial with respect to the mean (center of weight) of the
            # points. A conversion is made to make calculations simple. In case of the radial coordinates ("rt"), the
            # center point may be used as the reference, but then it might happen that it is placed outside the convex
            # hull of the points, so the polygonization algorithm would not work. Apart from that, this point might not
            # be near to the center of weight, which would cause that the algorithm would create a non-optimal (highly
            # concave) polygon.
            conversions = {
                "rt": lambda point: np.array([point[0] * np.cos(point[1]), point[0] * np.sin(point[1])])
            }
            if cls.coordinates_type in conversions:
                # TODO Test this conversion
                points = np.apply_along_axis(func1d=conversions[cls.coordinates_type], axis=1, arr=points)
            else:
                raise NotImplementedError("Conversion of {} coordinates into Cartesian coordinates is not implemented"
                                          "yet.".format(cls.coordinates_type))
        if np.all(points == points[0, :]):
            # Degenerate case of initializing mean of the distribution with a sequence of equal points
            # TODO Check, if this is the right thing to do
            return np.array([0])
        # TODO Maybe deal with other degenerate cases
        mean_point = np.mean(points, axis=0)
        points -= mean_point

        def to_polar_coordinates(point: np.ndarray) -> np.ndarray:
            x, y = point
            r = np.sqrt(x * x + y * y)
            if r == 0:
                t = 0
            else:
                angle = np.arccos(x / r)
                if y >= 0:
                    t = angle
                else:
                    t = 2 * np.pi - angle
            return np.array([r, t])

        points_polar = np.apply_along_axis(func1d=to_polar_coordinates, axis=1, arr=points)
        sorted_indices = np.argsort(points_polar[:, 1])
        points_polar = points_polar[sorted_indices]
        # TODO Maybe test better the following management of exotic cases
        # Remove duplicates (equal points)
        i = 0
        while i < sorted_indices.size - 1:
            j = i + 1
            while j < sorted_indices.size and points_polar[i, 1] == points_polar[j, 1]:
                j += 1
            j -= 1  # j is the index of the last point with the same t coordinate
            if j > i:
                # Some points have the same azimuthal angle
                k = i
                while k < j:
                    l = k + 1
                    while l <= j:
                        if points_polar[k, 0] == points_polar[l, 0]:
                            print("Warning: two points are equal, so one point is deleted")
                            sorted_indices = np.delete(arr=sorted_indices, obj=l)
                            points_polar = np.delete(arr=points_polar, obj=l, axis=0)
                            j -= 1
                        else:
                            l += 1
                    k += 1
                i = j + 1
            else:
                i += 1
        # Manage situations when multiple points have the same azimuthal angle
        i = 0
        while i < sorted_indices.size - 1:
            if points_polar[i, 1] == points_polar[i + 1, 1]:  # Maybe use np.isclose instead
                print("Information: some points have the same azimuthal angle")
                j = i + 2  # Finally j will be equal 1 + (the index of the last point with the same t coordinate)
                while j < sorted_indices.size and points_polar[i, 1] == points_polar[j, 1]:
                    j += 1
                if j - i > 2:
                    print("Warning: more than two points have the same azimuthal angle, so at least one point will not"
                          " be included in the polygonization, because then subsequent sides would be collinear")
                radii = points_polar[i:j, 0]
                min_radii_index = np.argmin(radii) + i
                max_radii_index = np.argmax(radii) + i
                if i == 0:
                    # From the point with the minimal r to the point with the maximal r (other option: compare somehow
                    # with the last point, but the last point may also have the same azimuthal angle as another point)
                    min_radii_point_index = sorted_indices[min_radii_index]
                    max_radii_point_index = sorted_indices[max_radii_index]
                    min_radii_point_polar = points_polar[min_radii_index]
                    max_radii_point_polar = points_polar[max_radii_index]
                    sorted_indices = np.delete(arr=sorted_indices, obj=np.s_[i:j])
                    points_polar = np.delete(arr=points_polar, obj=np.s_[i:j], axis=0)
                    sorted_indices = np.insert(arr=sorted_indices, obj=i, values=[min_radii_point_index,
                                                                                  max_radii_point_index])
                    points_polar = np.insert(arr=points_polar, obj=i, values=[min_radii_point_polar,
                                                                              max_radii_point_polar], axis=0)
                else:
                    # Choose a point from two points with minimal and maximal r that is closer to the previous point
                    # (with smaller t). Connect the previous point with this point, and this point with the other point
                    # of this two points
                    previous_point = points[sorted_indices[i - 1]]
                    indices = np.array([min_radii_index, max_radii_index])
                    # min_radii_point_index = sorted_indices[min_radii_index]
                    # max_radii_point_index = sorted_indices[max_radii_index]
                    # min_radii_point_polar = points_polar[min_radii_index]
                    # max_radii_point_polar = points_polar[max_radii_index]
                    # squared_distances = np.array([np.power(points[min_radii_point_index] - previous_point, 2),
                    #                               np.power(points[max_radii_point_index] - previous_point, 2)])
                    points_indices = sorted_indices[indices]
                    polars = points_polar[indices]
                    squared_distances = np.sum(np.power(points[points_indices] - previous_point, 2), axis=1)
                    sorted_indices = np.delete(arr=sorted_indices, obj=np.s_[i:j])
                    points_polar = np.delete(arr=points_polar, obj=np.s_[i:j], axis=0)
                    order = np.argsort(squared_distances)
                    sorted_indices = np.insert(arr=sorted_indices, obj=i, values=points_indices[order])
                    points_polar = np.insert(arr=points_polar, obj=i, values=polars[order], axis=0)
                # The next point (i + 1)-th need not to be checked, because it must have different t than the (i + 2)-th
                i += 2
            else:
                i += 1
        if points_polar[sorted_indices.size - 1, 1] == points_polar[0, 1]:
            print("Warning: all points are collinear")
            # Can logger be set as a class attribute, so that it would be accessible in the class methods?
        # Removing collinear points on the sides
        i = 1
        while i < sorted_indices.size - 1:
            if np.cross(points[sorted_indices[i]] - points[sorted_indices[i - 1]],
                        points[sorted_indices[i + 1]] - points[sorted_indices[i]]) == 0:
                print("Warning: two sides collinear, so one point is deleted")
                sorted_indices = np.delete(arr=sorted_indices, obj=i)
            else:
                i += 1
        if np.cross(points[sorted_indices[sorted_indices.size - 1]] - points[sorted_indices[sorted_indices.size - 2]],
                    points[sorted_indices[0]] - points[sorted_indices[sorted_indices.size - 1]]) == 0:
            print("Warning: two sides collinear, so one point is deleted")
            sorted_indices = np.delete(arr=sorted_indices, obj=sorted_indices.size - 1)
        if np.cross(points[sorted_indices[0]] - points[sorted_indices[sorted_indices.size - 1]],
                    points[sorted_indices[1]] - points[sorted_indices[0]]) == 0:
            print("Warning: two sides collinear, so one point is deleted")
            sorted_indices = np.delete(arr=sorted_indices, obj=0)
        return sorted_indices


class ConstrXYPolygonRSACMAESOpt(PolygonRSACMAESOpt, metaclass=abc.ABCMeta):

    default_rsa_parameters = dict(PolygonRSACMAESOpt.default_rsa_parameters,  # super().default_rsa_parameters,
                                  **{"maxVoxels": "4000000",
                                     "requestedAngularVoxelSize": "0.3",
                                     "minDx": "0.0",
                                     "from": "0",
                                     "collectors": "5",
                                     "split": "100000",
                                     "boundaryConditions": "periodic"})
    coordinates_type: str = "xy"

    def get_arg_signature(self) -> str:
        vertices_num = (self.initial_mean.size - 1) // 2 + 2
        return "vertices-" + str(vertices_num) + "-initstds-" + str(self.initial_stddevs)

    @classmethod
    def arg_to_points_coordinates(cls, arg: np.ndarray) -> np.ndarray:
        arg_with_all_coordinates = np.concatenate((arg, np.zeros(3))).reshape(-1, 2)
        return arg_with_all_coordinates

    @classmethod
    def points_coordinates_to_arg(cls, points: np.ndarray) -> np.ndarray:
        return points.flatten()[:-3]

    @classmethod
    def stddevs_to_points_stddevs(cls, arg: np.ndarray, stddevs: np.ndarray, covariance_matrix: np.ndarray) \
            -> np.ndarray:
        stddevs_with_all_coordinates = np.concatenate((stddevs, np.zeros(3))).reshape(-1, 2)
        return stddevs_with_all_coordinates


class ConstrXYConvexPolygonRSACMAESOpt(ConstrXYPolygonRSACMAESOpt, ConvexPolygonRSACMAESOpt):
    pass


class ConstrXYStarShapedPolygonRSACMAESOpt(ConstrXYPolygonRSACMAESOpt, StarShapedPolygonRSACMAESOpt):
    pass


class UniformTPolygonRSACMAESOpt(PolygonRSACMAESOpt, metaclass=abc.ABCMeta):

    min_radial_coordinate_optclattr: float = None
    max_radial_coordinate_optclattr: float = None
    rad_coord_trans_steepness_optclattr: float = None
    coordinates_type: str = "rt"

    def get_arg_signature(self) -> str:
        return "vertices-" + str(self.initial_mean.size - 1) + "-initstds-" + str(self.initial_stddevs) \
               + "-inr-" + str(self.min_radial_coordinate_optclattr) \
               + "-outr-" + str(self.max_radial_coordinate_optclattr)

    @classmethod
    def select_vertices(cls, points: np.ndarray) -> np.ndarray:
        return np.arange(points.shape[0])

    @classmethod
    def arg_to_points_coordinates(cls, arg: np.ndarray) -> np.ndarray:
        radial_coordinates = logistic(arg,
                                      cls.min_radial_coordinate_optclattr,
                                      cls.max_radial_coordinate_optclattr,
                                      cls.rad_coord_trans_steepness_optclattr)
        azimuthal_coordinates = np.linspace(start=0, stop=2 * np.pi, num=arg.size, endpoint=False)
        return np.stack((radial_coordinates, azimuthal_coordinates), axis=1)

    @classmethod
    def points_coordinates_to_arg(cls, points: np.ndarray) -> np.ndarray:
        # TODO Implement it
        pass


class RoundedPolygonRSACMAESOpt(PolygonRSACMAESOpt, metaclass=abc.ABCMeta):

    mode_rsa_parameters: dict = dict(RSACMAESOptimization.mode_rsa_parameters,
                                     particleType="RoundedPolygon")

    @classmethod
    @abc.abstractmethod
    def arg_to_radius_and_polygon_arg(cls, arg: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Function returning tuple containing radius of rounding of the polygon and the polygon argument based on the
        rounded polygon argument
        TODO Redact docstring
        """
        pass

    @classmethod
    @abc.abstractmethod
    def stddevs_to_radius_and_polygon_stddevs(cls, stddevs: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Function returning tuple containing rounding radius' standard deviation and the polygon argument's standard
        deviations based on the rounded polygon argument's standard deviations
        TODO Redact docstring
        """
        pass

    @classmethod
    def arg_to_particle_attributes(cls, arg: np.ndarray) -> str:
        """Function returning rsa3d program's parameter particleAttributes based on arg"""
        radius, polygon_arg = cls.arg_to_radius_and_polygon_arg(arg)
        polygon_particle_attributes = super().arg_to_particle_attributes(polygon_arg)
        if issubclass(cls, ConvexPolygonRSACMAESOpt):
            # If the polygon is convex, then don't pass its area
            return str(radius) + " " + polygon_particle_attributes
        # Extract particle data
        particle_attributes_list = polygon_particle_attributes.split(" ")
        vertices_num = int(particle_attributes_list[0])
        coordinates_type = particle_attributes_list[1]
        part_data = np.array(particle_attributes_list[2:2 + 2 * vertices_num],
                             dtype=np.float).reshape(-1, 2)
        if coordinates_type != "xy":
            conversions = {
                "rt": lambda point: np.array([point[0] * np.cos(point[1]), point[0] * np.sin(point[1])])
            }
            if coordinates_type in conversions:
                # TODO Test this conversion
                part_data = np.apply_along_axis(func1d=conversions[coordinates_type], axis=1, arr=part_data)
            else:
                raise NotImplementedError("Conversion of {} coordinates into Cartesian coordinates is not implemented"
                                          "yet.".format(coordinates_type))
        if ConvexHull(part_data).vertices.size == vertices_num:
            # If the polygon is convex, then don't pass its area
            return str(radius) + " " + polygon_particle_attributes
        else:
            # Polygon is concave, calculate and pass its area
            # Method of calculation valid for simple polygons
            polygon = shapely.geometry.Polygon(shell=part_data)
            rounded_polygon = polygon.buffer(distance=radius, resolution=10 ** 6)
            # For the resolution of 10^6, the relative error for calculation of the unitary disk area approximately
            # equals 4.0 * 10^-13 and the time of this calculation is a few seconds
            area = rounded_polygon.area
            return " ".join([str(radius), polygon_particle_attributes, str(area)])

    @classmethod
    def stddevs_to_particle_stddevs(cls, arg: np.ndarray, stddevs: np.ndarray, covariance_matrix: np.ndarray) \
            -> np.ndarray:
        radius_stddev, polygon_stddevs = cls.stddevs_to_radius_and_polygon_stddevs(stddevs)
        polygon_arg = cls.arg_to_radius_and_polygon_arg(arg)[1]
        polygon_covariance_matrix = covariance_matrix[1:, 1:]  # TODO Maybe create a method for that
        polygon_particle_stddevs = super().stddevs_to_particle_stddevs(polygon_arg,
                                                                       polygon_stddevs,
                                                                       polygon_covariance_matrix)
        return np.insert(polygon_particle_stddevs, 0, radius_stddev)

    @classmethod
    def draw_particle(cls, particle_attributes: str, scaling_factor: float, color: str,
                      arg: Optional[np.ndarray] = None, std_devs: Optional[np.ndarray] = None,
                      covariance_matrix: Optional[np.ndarray] = None, part_std_devs: Optional[np.ndarray] = None) \
            -> matplotlib.offsetbox.DrawingArea:
        # Extract particle data
        particle_attributes_list = particle_attributes.split(" ")
        radius = float(particle_attributes_list[0])
        vertices_num = int(particle_attributes_list[1])
        coordinates_type = particle_attributes_list[2]
        part_data = np.array(particle_attributes_list[3:3 + 2 * vertices_num],
                             dtype=np.float).reshape(-1, 2)
        if coordinates_type != "xy":
            conversions = {
                "rt": lambda point: np.array([point[0] * np.cos(point[1]), point[0] * np.sin(point[1])])
            }
            if coordinates_type in conversions:
                # TODO Test this conversion
                part_data = np.apply_along_axis(func1d=conversions[coordinates_type], axis=1, arr=part_data)
            else:
                raise NotImplementedError("Conversion of {} coordinates into Cartesian coordinates is not implemented"
                                          "yet.".format(coordinates_type))
        if np.all(part_data == part_data[0, :]):
            # Degenerate case of initializing mean of the distribution with a sequence of equal points
            sqrt_area = np.sqrt(np.pi) * radius
            disk_center = part_data[0, :] / sqrt_area
            radius /= sqrt_area
            if part_std_devs is None:
                drawing_area = matplotlib.offsetbox.DrawingArea(scaling_factor * 2 * radius,
                                                                scaling_factor * 2 * radius,
                                                                scaling_factor * -(disk_center[0] - radius),
                                                                scaling_factor * -(disk_center[1] - radius))
            else:
                polygon_arg = cls.arg_to_radius_and_polygon_arg(arg)[1]
                radius_std_dev, polygon_std_devs = cls.stddevs_to_radius_and_polygon_stddevs(std_devs)
                polygon_covariance_matrix = covariance_matrix[1:, 1:]  # TODO Maybe create a method for that
                points_std_devs = cls.stddevs_to_points_stddevs(polygon_arg,
                                                                polygon_std_devs,
                                                                polygon_covariance_matrix)
                # TODO t coordinate's (angle's) standard deviation should not be scaled
                points_std_devs_data = points_std_devs.reshape(-1, 2) / sqrt_area
                radius_std_dev /= sqrt_area
                # TODO Maybe add drawing rounding radius' standard deviation
                max_x = np.max(np.append(points_std_devs_data[:, 0], radius))
                max_y = np.max(np.append(points_std_devs_data[:, 1], radius))
                drawing_area = matplotlib.offsetbox.DrawingArea(scaling_factor * 2 * max_x,
                                                                scaling_factor * 2 * max_y,
                                                                scaling_factor * -(disk_center[0] - max_x),
                                                                scaling_factor * -(disk_center[1] - max_y))
            disk = matplotlib.patches.Circle((scaling_factor * disk_center[0], scaling_factor * disk_center[1]),
                                              scaling_factor * radius,
                                              color=color)
            drawing_area.add_artist(disk)
            if part_std_devs is not None:
                arrow_style = matplotlib.patches.ArrowStyle("|-|", widthA=0, widthB=1.0)
                center = (scaling_factor * disk_center[0], scaling_factor * disk_center[1])
                if coordinates_type == "xy":
                    for point_std_dev in points_std_devs_data:
                        ticks = [(center[0] + scaling_factor * point_std_dev[0], center[1]),
                                 (center[0] - scaling_factor * point_std_dev[0], center[1]),
                                 (center[0], center[1] + scaling_factor * point_std_dev[1]),
                                 (center[0], center[1] - scaling_factor * point_std_dev[1])]
                        for tick in ticks:
                            std_dev_arrow = matplotlib.patches.FancyArrowPatch(
                                center,
                                tick,
                                arrowstyle=arrow_style,
                                shrinkA=0,
                                shrinkB=0)
                            drawing_area.add_artist(std_dev_arrow)
                elif coordinates_type == "rt":
                    center_r = np.sqrt(center[0] * center[0] + center[1] * center[1])
                    for point_std_dev in points_std_devs_data:
                        arrow_r = (scaling_factor * point_std_dev[0] * center[0] / center_r,
                                   scaling_factor * point_std_dev[0] * center[1] / center_r)
                        arrow_t = (scaling_factor * point_std_dev[1] * center[1] / center_r,
                                   -scaling_factor * point_std_dev[1] * center[0] / center_r)
                        ticks = [(center[0] + arrow_r[0], center[1] + arrow_r[1]),
                                 (center[0] - arrow_r[0], center[1] - arrow_r[1]),
                                 (center[0] + arrow_t[0], center[1] + arrow_t[1]),
                                 (center[0] - arrow_t[0], center[1] - arrow_t[1])]
                        for tick in ticks:
                            std_dev_arrow = matplotlib.patches.FancyArrowPatch(
                                center,
                                tick,
                                arrowstyle=arrow_style,
                                shrinkA=0,
                                shrinkB=0)
                            drawing_area.add_artist(std_dev_arrow)
            return drawing_area
        # Calculate particle area
        if issubclass(cls, ConvexPolygonRSACMAESOpt) or ConvexHull(part_data).vertices.size == vertices_num:
            # Method of calculation valid for convex polygons
            # TODO Maybe add a method calculating particle's area
            center_of_mass = np.mean(part_data, axis=0)
            area = 0.
            for vert_num in range(vertices_num):
                prev_vert_num = vert_num - 1 if vert_num > 0 else vertices_num - 1
                next_vert_num = (vert_num + 1) % vertices_num
                area += np.abs(np.cross(part_data[vert_num] - center_of_mass,
                                        part_data[next_vert_num] - center_of_mass)) / 2
                first_segment_vec = part_data[prev_vert_num] - part_data[vert_num]
                second_segment_vec = part_data[next_vert_num] - part_data[vert_num]
                triangle_side_vec = part_data[next_vert_num] - part_data[prev_vert_num]
                triangle_height = np.abs(np.cross(first_segment_vec, second_segment_vec)) \
                    / np.linalg.norm(triangle_side_vec)
                angle = np.arccos(triangle_height / np.linalg.norm(first_segment_vec)) \
                    + np.arccos(triangle_height / np.linalg.norm(second_segment_vec))
                area += radius * (np.linalg.norm(first_segment_vec) + radius * (np.pi - angle) / 2.)
        else:
            # Area is given in the particleAttributes parameter
            area = float(particle_attributes_list[-1])
        sqrt_area = np.sqrt(area)
        part_data /= sqrt_area
        radius /= sqrt_area
        if arg is not None:
            polygon_arg = cls.arg_to_radius_and_polygon_arg(arg)[1]
            points_coordinates = cls.arg_to_points_coordinates(polygon_arg)
            # coordinates_type is overwritten, although it should be the same
            if coordinates_type != "xy":
                conversions = {
                    "rt": lambda point: np.array([point[0] * np.cos(point[1]), point[0] * np.sin(point[1])])
                }
                if coordinates_type in conversions:
                    # TODO Test this conversion
                    points_coordinates = np.apply_along_axis(func1d=conversions[coordinates_type],
                                                             axis=1,
                                                             arr=points_coordinates)
                else:
                    raise NotImplementedError("Conversion of {} coordinates into Cartesian coordinates is not"
                                              "implemented yet.".format(coordinates_type))
            points_data = points_coordinates.reshape(-1, 2) / sqrt_area
        if part_std_devs is not None:
            polygon_arg = cls.arg_to_radius_and_polygon_arg(arg)[1]
            radius_std_dev, polygon_std_devs = cls.stddevs_to_radius_and_polygon_stddevs(std_devs)
            polygon_covariance_matrix = covariance_matrix[1:, 1:]  # TODO Maybe create a method for that
            points_std_devs = cls.stddevs_to_points_stddevs(polygon_arg, polygon_std_devs, polygon_covariance_matrix)
            points_std_devs_data = points_std_devs.reshape(-1, 2) / sqrt_area
            # std_devs_data = part_std_devs.reshape(-1, 2) / sqrt_area
            radius_std_dev /= sqrt_area
            # TODO Maybe add drawing rounding radius' standard deviation
        # Draw particle
        # Get polygon drawing's width and height
        if part_std_devs is None:
            shown_points_data = points_data if arg is not None else part_data
            x_min = np.min(shown_points_data[:, 0] - radius)
            x_max = np.max(shown_points_data[:, 0] + radius)
            y_min = np.min(shown_points_data[:, 1] - radius)
            y_max = np.max(shown_points_data[:, 1] + radius)
        else:
            # If part_std_devs are given, arg and std_devs should also be given
            if coordinates_type == "xy":
                x_min = np.min(np.concatenate((points_data[:, 0] - radius,
                                               points_data[:, 0] - points_std_devs_data[:, 0])))
                x_max = np.max(np.concatenate((points_data[:, 0] + radius,
                                               points_data[:, 0] + points_std_devs_data[:, 0])))
                y_min = np.min(np.concatenate((points_data[:, 1] - radius,
                                               points_data[:, 1] - points_std_devs_data[:, 1])))
                y_max = np.max(np.concatenate((points_data[:, 1] + radius,
                                               points_data[:, 1] + points_std_devs_data[:, 1])))
            elif coordinates_type == "rt":
                arrows_list = []
                for point_num, point in enumerate(points_data):
                    point_r = np.sqrt(point[0] * point[0] + point[1] * point[1])
                    arrow_r = (points_std_devs_data[point_num][0] * point[0] / point_r,
                               points_std_devs_data[point_num][0] * point[1] / point_r)
                    arrow_t = (points_std_devs_data[point_num][1] * point[1] / point_r,
                               -points_std_devs_data[point_num][1] * point[0] / point_r)
                    ticks = [(point[0] + arrow_r[0], point[1] + arrow_r[1]),
                             (point[0] - arrow_r[0], point[1] - arrow_r[1]),
                             (point[0] + arrow_t[0], point[1] + arrow_t[1]),
                             (point[0] - arrow_t[0], point[1] - arrow_t[1])]
                    arrows_list.extend(ticks)
                arrows = np.array(arrows_list)

                x_min = np.min(np.concatenate((points_data[:, 0] - radius, arrows[:, 0])))
                x_max = np.max(np.concatenate((points_data[:, 0] + radius, arrows[:, 0])))
                y_min = np.min(np.concatenate((points_data[:, 1] - radius, arrows[:, 1])))
                y_max = np.max(np.concatenate((points_data[:, 1] + radius, arrows[:, 1])))
        drawing_area = matplotlib.offsetbox.DrawingArea(scaling_factor * (x_max - x_min),
                                                        scaling_factor * (y_max - y_min),
                                                        scaling_factor * -x_min,
                                                        scaling_factor * -y_min)
        # TODO Check if the scale of the radius is correct - rather yes
        # TODO Check why a strange artefact appeared
        polygon = matplotlib.patches.Polygon(scaling_factor * part_data, linewidth=scaling_factor * 2 * radius,
                                             joinstyle="round", capstyle="round", color=color)
        drawing_area.add_artist(polygon)
        if part_std_devs is None:
            pass
            # for point_num, point_args in enumerate(shown_points_data):
            #     is_vertex = np.any([np.allclose(point_args, vertex_args) for vertex_args in part_data])
            #     point_label = matplotlib.text.Text(x=scaling_factor * point_args[0],
            #                                        y=scaling_factor * point_args[1],
            #                                        text=str(point_num),
            #                                        horizontalalignment="center",
            #                                        verticalalignment="center",
            #                                        fontsize=11 if is_vertex else 9,
            #                                        fontweight="normal" if is_vertex else "bold")
            #     drawing_area.add_artist(point_label)
        else:
            if coordinates_type == "xy":
                for point_num, point_args in enumerate(points_data):
                    # point_label = matplotlib.text.Text(x=scaling_factor * point_args[0] + scaling_factor / 10,
                    #                                    y=scaling_factor * point_args[1] + scaling_factor / 10,
                    #                                    text=str(point_num),
                    #                                    horizontalalignment="center",
                    #                                    verticalalignment="center",
                    #                                    fontsize=9)
                    # drawing_area.add_artist(point_label)
                    # TODO Maybe add dots marking the positions of the points, especially the point(s) with 0 standard
                    #  deviations

                    # arrow_style = matplotlib.patches.ArrowStyle("->", head_length=0.)
                    arrow_style = matplotlib.patches.ArrowStyle("|-|", widthA=0, widthB=1.0)
                    # arrow_style = matplotlib.patches.ArrowStyle("simple", head_width=1.2)  # Causes a bug in matplotlib
                    # arrow_style = matplotlib.patches.ArrowStyle("->", head_width=0.8)
                    # Head lengths are not scaled and for small standard deviations heads are longer than arrow, so one
                    # solution is to make them not visible
                    # TODO Make arrows lengths correct while using arrows without heads
                    center = (scaling_factor * point_args[0], scaling_factor * point_args[1])
                    ticks = [(center[0] + scaling_factor * points_std_devs_data[point_num][0], center[1]),
                             (center[0] - scaling_factor * points_std_devs_data[point_num][0], center[1]),
                             (center[0], center[1] + scaling_factor * points_std_devs_data[point_num][1]),
                             (center[0], center[1] - scaling_factor * points_std_devs_data[point_num][1])]
                    for tick in ticks:
                        std_dev_arrow = matplotlib.patches.FancyArrowPatch(
                            center,
                            tick,
                            arrowstyle=arrow_style,
                            shrinkA=0,
                            shrinkB=0)
                        drawing_area.add_artist(std_dev_arrow)
            elif coordinates_type == "rt":
                # TODO Make arrows lengths correct while using arrows without heads
                for point_num, point_args in enumerate(points_data):
                    # point_label = matplotlib.text.Text(x=scaling_factor * point_args[0] + scaling_factor / 10,
                    #                                    y=scaling_factor * point_args[1] + scaling_factor / 10,
                    #                                    text=str(point_num),
                    #                                    horizontalalignment="center",
                    #                                    verticalalignment="center",
                    #                                    fontsize=9)
                    # drawing_area.add_artist(point_label)
                    # TODO Maybe add dots marking the positions of the points, especially the point(s) with 0 standard
                    #  deviations
                    disk = matplotlib.patches.Circle(
                        (scaling_factor * point_args[0], scaling_factor * point_args[1]),
                        scaling_factor * 0.03,
                        color="k")
                    drawing_area.add_artist(disk)

                    arrow_style = matplotlib.patches.ArrowStyle("|-|", widthA=0, widthB=1.0)
                    center = (scaling_factor * point_args[0], scaling_factor * point_args[1])
                    for tick in scaling_factor * arrows[4 * point_num:4 * point_num + 4]:
                        std_dev_arrow = matplotlib.patches.FancyArrowPatch(
                            center,
                            tick,
                            arrowstyle=arrow_style,
                            shrinkA=0,
                            shrinkB=0)
                        drawing_area.add_artist(std_dev_arrow)
                    
        return drawing_area


class FixedRadiiRoundedPolygonRSACMAESOpt(RoundedPolygonRSACMAESOpt, metaclass=abc.ABCMeta):

    @classmethod
    def arg_to_radius_and_polygon_arg(cls, arg: np.ndarray) -> Tuple[float, np.ndarray]:
        return 1, arg

    @classmethod
    def stddevs_to_radius_and_polygon_stddevs(cls, stddevs: np.ndarray) -> Tuple[float, np.ndarray]:
        return 0, stddevs


class VariableRadiiRoundedPolygonRSACMAESOpt(RoundedPolygonRSACMAESOpt, metaclass=abc.ABCMeta):

    @classmethod
    def arg_to_radius_and_polygon_arg(cls, arg: np.ndarray) -> Tuple[float, np.ndarray]:
        return softplus(arg[0]), arg[1:]

    @classmethod
    def stddevs_to_radius_and_polygon_stddevs(cls, stddevs: np.ndarray) -> Tuple[float, np.ndarray]:
        # TODO Correct this transformation
        return softplus(stddevs[0]), stddevs[1:]


class ConstrXYFixedRadiiRoundedConvexPolygonRSACMAESOpt(FixedRadiiRoundedPolygonRSACMAESOpt,
                                                        ConstrXYConvexPolygonRSACMAESOpt):
    pass


class ConstrXYFixedRadiiRoundedStarShapedPolygonRSACMAESOpt(FixedRadiiRoundedPolygonRSACMAESOpt,
                                                            ConstrXYStarShapedPolygonRSACMAESOpt):
    pass


class ConstrXYVariableRadiiRoundedConvexPolygonRSACMAESOpt(VariableRadiiRoundedPolygonRSACMAESOpt,
                                                           ConstrXYConvexPolygonRSACMAESOpt):
    pass


class ConstrXYVariableRadiiRoundedStarShapedPolygonRSACMAESOpt(VariableRadiiRoundedPolygonRSACMAESOpt,
                                                               ConstrXYStarShapedPolygonRSACMAESOpt):
    pass


class VariableRadiiRoundedUniformTPolygonRSACMAESOpt(VariableRadiiRoundedPolygonRSACMAESOpt,
                                                     UniformTPolygonRSACMAESOpt):
    pass


def optimize() -> None:

    def opt_fixed_radii() -> None:
        initial_mean = np.zeros(2 * optimization_input["opt_mode_args"]["disks_num"])
        opt_class_args = dict(optimization_input["opt_class_args"])  # Use dict constructor to copy by value
        opt_class_args["initial_mean"] = initial_mean
        opt_class_args["optimization_input"] = optimization_input
        fixed_radii_polydisk_opt = FixedRadiiXYPolydiskRSACMAESOpt(**opt_class_args)
        fixed_radii_polydisk_opt.run()

    def opt_constr_fixed_radii() -> None:
        initial_mean = np.zeros(2 * optimization_input["opt_mode_args"]["disks_num"] - 3)
        opt_class_args = dict(optimization_input["opt_class_args"])  # Use dict constructor to copy by value
        opt_class_args["initial_mean"] = initial_mean
        opt_class_args["optimization_input"] = optimization_input
        constr_fixed_radii_polydisk_opt = ConstrFixedRadiiXYPolydiskRSACMAESOpt(**opt_class_args)
        constr_fixed_radii_polydisk_opt.run()

    def opt_constr_fixed_radii_convex_polygon() -> None:
        initial_mean = np.zeros(2 * optimization_input["opt_mode_args"]["vertices_num"] - 3)
        opt_class_args = dict(optimization_input["opt_class_args"])  # Use dict constructor to copy by value
        opt_class_args["initial_mean"] = initial_mean
        opt_class_args["optimization_input"] = optimization_input
        constr_fixed_radii_polygon_opt = ConstrXYFixedRadiiRoundedConvexPolygonRSACMAESOpt(**opt_class_args)
        constr_fixed_radii_polygon_opt.run()

    def opt_constr_fixed_radii_star_shaped_polygon() -> None:
        if optimization_input["opt_mode_args"]["initial_mean"] == "origin":
            initial_mean = np.zeros(2 * optimization_input["opt_mode_args"]["vertices_num"] - 3)
        elif optimization_input["opt_mode_args"]["initial_mean"] == "regular_polygon":
            vertices_num = optimization_input["opt_mode_args"]["vertices_num"]
            radius = optimization_input["opt_mode_args"]["initial_mean_params"]["radius"]
            angles = np.pi * (3 / 2 - np.arange(start=3, stop=2 * vertices_num + 2, step=2) / vertices_num)
            vertices_centered = np.apply_along_axis(func1d=lambda angle: np.array([radius * np.cos(angle),
                                                                                   radius * np.sin(angle)]),
                                                    axis=0,
                                                    arr=angles).T
            shift_angle = np.pi * (1 / 2 - 1 / vertices_num)
            vertices = vertices_centered + np.array([radius * np.cos(shift_angle), radius * np.sin(shift_angle)])
            initial_mean = vertices.flatten()[:-3]
        opt_class_args = dict(optimization_input["opt_class_args"])  # Use dict constructor to copy by value
        opt_class_args["initial_mean"] = initial_mean
        opt_class_args["optimization_input"] = optimization_input
        constr_fixed_radii_polygon_opt = ConstrXYFixedRadiiRoundedStarShapedPolygonRSACMAESOpt(**opt_class_args)
        constr_fixed_radii_polygon_opt.run()

    def opt_constr_variable_radii_star_shaped_polygon() -> None:
        if optimization_input["opt_mode_args"]["polygon_initial_mean"] == "origin":
            polygon_initial_mean = np.zeros(2 * optimization_input["opt_mode_args"]["vertices_num"] - 3)
        elif optimization_input["opt_mode_args"]["polygon_initial_mean"] == "regular_polygon":
            vertices_num = optimization_input["opt_mode_args"]["vertices_num"]
            polygon_radius = optimization_input["opt_mode_args"]["initial_mean_params"]["polygon_radius"]
            angles = np.pi * (3 / 2 - np.arange(start=3, stop=2 * vertices_num + 2, step=2) / vertices_num)
            vertices_centered = np.apply_along_axis(func1d=lambda angle: np.array([polygon_radius * np.cos(angle),
                                                                                   polygon_radius * np.sin(angle)]),
                                                    axis=0,
                                                    arr=angles).T
            shift_angle = np.pi * (1 / 2 - 1 / vertices_num)
            vertices = vertices_centered + np.array([polygon_radius * np.cos(shift_angle),
                                                     polygon_radius * np.sin(shift_angle)])
            polygon_initial_mean = vertices.flatten()[:-3]
        initial_mean = np.insert(polygon_initial_mean, 0, optimization_input["opt_mode_args"]["rounding_initial_mean"])
        opt_class_args = dict(optimization_input["opt_class_args"])  # Use dict constructor to copy by value
        opt_class_args["initial_mean"] = initial_mean
        opt_class_args["optimization_input"] = optimization_input
        constr_variable_radii_polygon_opt = ConstrXYVariableRadiiRoundedStarShapedPolygonRSACMAESOpt(**opt_class_args)
        constr_variable_radii_polygon_opt.run()

    def opt_variable_radii_uniform_t_polygon() -> None:
        if optimization_input["opt_mode_args"]["polygon_initial_mean"] == "regular_polygon":
            polygon_initial_mean = np.full(shape=optimization_input["opt_mode_args"]["vertices_num"],
                                           fill_value=optimization_input["opt_mode_args"]["initial_mean_params"]
                                               ["polygon_radius"])
        initial_mean = np.insert(polygon_initial_mean, 0, optimization_input["opt_mode_args"]["rounding_initial_mean"])
        opt_class_args = dict(optimization_input["opt_class_args"])  # Use dict constructor to copy by value
        opt_class_args["initial_mean"] = initial_mean
        opt_class_args["optimization_input"] = optimization_input
        constr_fixed_radii_polygon_opt = VariableRadiiRoundedUniformTPolygonRSACMAESOpt(**opt_class_args)
        constr_fixed_radii_polygon_opt.run()


    opt_modes = {
        "optfixedradii": opt_fixed_radii,
        "optconstrfixedradii": opt_constr_fixed_radii,
        "optconstrfixedradiiconvexpolygon": opt_constr_fixed_radii_convex_polygon,
        "optconstrfixedradiistarshapedpolygon": opt_constr_fixed_radii_star_shaped_polygon,
        "optconstrvariableradiistarshapedpolygon": opt_constr_variable_radii_star_shaped_polygon,
        "optvariableradiiuniformtpolygon": opt_variable_radii_uniform_t_polygon,
    }
    if args.file is None:
        raise TypeError("In optimize mode input file has to be specified using -f argument")
    with open(_input_dir + "/" + args.file, "r") as opt_input_file:
        # TODO Maybe use configparser module or YAML format instead
        optimization_input = json.load(opt_input_file)
    opt_modes[optimization_input["opt_mode"]]()


def plot_cmaes_optimization_data() -> None:
    if args.signature is None:
        raise TypeError("In plotcmaesoptdata mode optimization signature has to be specified using -s argument")
    opt_class_name = args.signature.split("-")[5]
    # Get optimization class from current module.
    # If the class is not in current module, module's name has to be passed as sys.modules dictionary's key,
    # so such classes should put the module name to optimization signature.
    opt_class = getattr(sys.modules[__name__], opt_class_name)
    opt_class.set_optimization_class_attributes(signature=args.signature)
    config_file_name = args.config if args.config is not None else "graph_config.yaml"
    opt_class.plot_optimization_data(signature=args.signature, config_file_name=config_file_name)


def resume_optimization() -> None:
    if args.signature is None:
        raise TypeError("In resumeoptimization mode optimization signature has to be specified using -s argument")
    opt_class_name = args.signature.split("-")[5]
    # Get optimization class from current module.
    # If the class is not in current module, module's name has to be passed as sys.modules dictionary's key,
    # so such classes should put the module name to optimization signature. Such a class also needs to be explicitly
    # imported before unpickling.
    opt_class = getattr(sys.modules[__name__], opt_class_name)
    # Optimization directory has to be prepared - e.g. by duplicating original optimization directory and adding
    # "-restart-1" suffix at the end of the directory name, then (maybe it is necessary - check it) removing directories
    # in outrsa subdirectory corresponding to simulations in interrupted generation, maybe also removing some entries in
    # files in outcmaes subdirectory (check, if the call to self.CMAES.logger.add at the beginning of run method won't
    # spoil anything (it will cause duplicated CMA-ES generation data records).
    # But many classes' attributes depend on optimization directory - that's why it is better currently to duplicate
    # original optimization directory and add "-original" suffix at the end of the copied directory name and resume
    # optimization in original directory
    optimization = opt_class.unpickle(args.signature)
    optimization.logger.info(msg="")
    optimization.logger.info(msg="")
    optimization.logger.info(msg="Resuming optimization")
    interrupted_simulations_dirs = glob.glob(optimization.rsa_output_dir + "/{}_*".format(optimization.CMAES.countiter))
    if len(interrupted_simulations_dirs) > 0:
        optimization.logger.info(msg="Moving interrupted generation simulations' directories {} to the unused"
                                     " simulations folder".format(", ".join(map(os.path.basename,
                                                                                interrupted_simulations_dirs))))
        try:
            unused_simulations_dir = optimization.rsa_output_dir + "/unused_simulations"
            if not os.path.exists(unused_simulations_dir):
                os.makedirs(unused_simulations_dir)
            for directory in interrupted_simulations_dirs:
                shutil.move(directory, unused_simulations_dir)
        except Exception as exception:
            optimization.logger.warning(msg="Exception raised when moving interrupted generation simulations'"
                                            " directories; {}: {}\n{}".format(type(exception).__name__, exception,
                                                                              traceback.format_exc(limit=6).strip()))
            # TODO Deal with this error, check what would happen if the directories weren't (re)moved, maybe use another
            #  way to move directories
    # Set optimization class attributes
    optimization.set_optimization_class_attributes(optimization_input=optimization.optimization_input)
    # Overwrite optimization options, if the file argument was given
    if args.file is not None:
        with open(_input_dir + "/" + args.file, "r") as opt_input_file:
            # TODO Maybe use configparser module or YAML format instead
            resume_input = json.load(opt_input_file)
        # TODO Test it
        if "cma_options" in resume_input:
            # After unpickling output is redirected to logger, so CMAEvolutionStrategy classes' errors and warnings
            # as e.g. "UserWarning: key popsize ignored (not recognized as versatile) ..." will be logged
            optimization.CMAES.opts.set(resume_input["cma_options"])
            # optimization.cma_options is not updated
        if "rsa_parameters" in resume_input:
            for param in list(optimization.mode_rsa_parameters) + ["particleAttributes"]:
                if param in resume_input["rsa_parameters"]:
                    del resume_input["rsa_parameters"][param]
                    optimization.logger.warning(msg="Resume RSA parameter {} ignored".format(param))
            optimization.rsa_parameters.update(resume_input["rsa_parameters"])
            if not optimization.input_given:
                optimization.all_rsa_parameters.update(resume_input["rsa_parameters"])
        for attr in ["accuracy", "parallel", "particle_attributes_parallel", "okeanos", "max_nodes_number",
                     "okeanos_parallel", "nodes_number", "collectors_per_task"]:
            if attr in resume_input:
                setattr(optimization, attr, resume_input[attr])
        if "min_collectors_number" in resume_input:
            optimization.min_collectors_number = max(resume_input["min_collectors_number"], 2)
        if "threads" in resume_input:
            optimization.parallel_threads_number = resume_input["threads"]
        if any(attr in resume_input for attr in ["threads", "okeanos", "max_nodes_number"]):
            if not optimization.okeanos:
                optimization.parallel_simulations_number = min(optimization.parallel_threads_number,
                                                               optimization.CMAES.popsize)
            else:
                optimization.parallel_simulations_number = min(optimization.max_nodes_number - 1,
                                                               optimization.CMAES.popsize) \
                    if optimization.max_nodes_number is not None else optimization.CMAES.popsize
        # TODO Set (and maybe check) other attributes, if needed
        if ("rsa_parameters" in resume_input and len(resume_input["rsa_parameters"]) > 0) or "accuracy" in resume_input\
                or "okeanos_parallel" in resume_input:
            # All attributes that are used in optimization.rsa_proc_arguments have to be set already, if present
            optimization.set_rsa_proc_arguments()
        # Generate used optimization input file in output directory
        resume_signature = datetime.datetime.now().isoformat(timespec="milliseconds").replace(":", "-")\
            .replace(".", "_")
        resume_signature += "-optimization-resume-gen-{}".format(optimization.CMAES.countiter)
        opt_input_filename = optimization.output_dir + "/" + resume_signature + "-input.json"
        with open(opt_input_filename, "w+") as opt_input_file:
            json.dump(resume_input, opt_input_file, indent=2)
        optimization.logger.info(msg="Optimization resume input file: {}-input.json".format(resume_signature))
        if "rsa_parameters" in resume_input and len(resume_input["rsa_parameters"]) > 0:
            rsa_input_filename = optimization.output_dir + "/" + resume_signature + "-rsa-input.txt"
            with open(rsa_input_filename, "w+") as rsa_input_file:
                # TODO Maybe use resume_input["rsa_parameters"] instead
                rsa_parameters = optimization.rsa_parameters if optimization.input_given\
                    else optimization.all_rsa_parameters
                rsa_input_file.writelines(["{} = {}\n".format(param_name, param_value)
                                           for param_name, param_value in rsa_parameters.items()])
            optimization.logger.info(msg="Resume RSA input file: {}-rsa-input.txt".format(resume_signature))
    # If optimization is to be run on Okeanos in parallel mode, try to set nodes_number attribute to the number of nodes
    # actually allocated to the SLURM job, unless nodes_number was given in resume input file
    if optimization.okeanos_parallel and (args.file is None or "nodes_number" not in resume_input):
        slurm_job_num_nodes = os.getenv("SLURM_JOB_NUM_NODES")
        if slurm_job_num_nodes is not None:
            optimization.nodes_number = int(slurm_job_num_nodes)
        else:
            optimization.logger.warning(msg="Unable to get number of nodes allocated to the job; SLURM_JOB_NUM_NODES"
                                            " environment variable is not set")
            if optimization.nodes_number is not None:
                optimization.logger.warning(msg="Using the previously set value of the nodes_number attribute")
            else:
                optimization.logger.warning(msg="Setting the value of the nodes_number attribute to 1 + (population"
                                                " size)")
                optimization.nodes_number = 1 + optimization.CMAES.popsize
    # Run optimization
    optimization.run()


# TODO Maybe read optimization directories' and files' names from a YAML file and use pathlib module
# TODO Add managing errors of the rsa3d program and forcing recalculation (or max n recalculations) or resampling of
#  the parameter point (maybe return None immediately after rsa_process.wait() in case of a failure) (old note)
# TODO Maybe prepare a Makefile like in https://docs.python-guide.org/writing/structure/ and with creating
#  virtual environment (check how PyCharm creates virtualenvs)
# TODO Maybe other methods for plotting and visualization of the saved data
# TODO Maybe add a method calling rsa3d program in wolfram mode on specified .bin file
#  - particleAttributes may be taken from rsa-simulation-input.txt file
# TODO Adding box constraints based on feasible anisotropy (probably)
# TODO Does writing to the single packing-fraction-vs-params.txt file by paralleled processes pose a problem?
# TODO Maybe decreasing packing fraction error in subsequent generations by accuracy mode
#  or increasing the number of collectors. Maybe combine that with decreasing the population.
# TODO Algorithms for noisy optimization: UH-CMA-ES (cma.NoiseHandler)? DX-NES-UE?
# TODO Maybe single, big collectors and uncertainty handling (variable numbers of reevaluations,
#  thus variable numbers of collectors - UH-CMA-ES)?
# TODO Does storing data affect performance?
# TODO Think, if demanding given accuracy (rsa3d accuracy mode) is the right thing to do
# TODO Check, if rsa3d options are well chosen (especially split) and wonder, if they should be somehow
#  automatically adjusted during optimization
if __name__ == '__main__':
    module_description = "Optimization of packing fraction of two-dimensional Random Sequential Adsorption (RSA)" \
                         " packings\nusing Covariance Matrix Adaptation Evolution Strategy (CMA-ES)."
    # TODO Maybe automate adding modes, for example using (parametrized) decorators
    module_modes = {
        "testcma": test_cma_package,
        "examplecmaplots": example_cma_plots,
        "optimize": optimize,
        # TODO Maybe do it in another way
        "plotcmaesoptdata": plot_cmaes_optimization_data,
        # TODO Test and improve it
        "resumeoptimization": resume_optimization
    }
    arg_parser = argparse.ArgumentParser(description=module_description)
    # TODO Maybe use sub-commands for modes (arg_parser.add_subparsers)
    arg_parser.add_argument("mode", choices=list(module_modes), help="program mode")
    # TODO Make this argument obligatory for optimize mode
    arg_parser.add_argument("-f", "--file", help="json input file from ./input directory")
    # TODO Make this argument obligatory for plotcmaesoptdata mode
    arg_parser.add_argument("-s", "--signature", help="optimization signature - name of subdirectory of ./output")
    # TODO Make this argument available only in plotcmaesoptdata mode
    # arg_parser.add_argument("-m", "--modulo", help="Annotate points with particles drawings in first, last and every"
    #                                                " modulo generation. If not given,"
    #                                                " modulo will be automatically adjusted.")
    # TODO Make this argument available only in plotcmaesoptdata mode
    arg_parser.add_argument("-c", "--config", help="name of graph configuration YAML file from optimization directory")
    args = arg_parser.parse_args()
    module_modes[args.mode]()
