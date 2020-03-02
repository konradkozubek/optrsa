"""Main module in optrsa project"""
# TODO Read about docstrings and automatic documentation generation services like pydoctor

import cma

import sys
import matplotlib
# PyCharm Professional sets other backend, which may cause trouble
if sys.platform.startswith('darwin'):  # MacOS
    # See https://matplotlib.org/tutorials/introductory/usage.html#backends
    matplotlib.use("MacOSX")
    # Qt5Agg, Qt4Agg - can't import qt bindings, GTK3Agg, GTK3Cairo - can't install all dependencies. nbAgg - fails.
    # WX does not have access to the screen. TkAgg works, WebAgg (with tornado imported) works worse than TkAgg.
else:
    # Partially tested
    matplotlib.use("Qt5Agg")  # Maybe try also TkAgg (works) if interactivity is needed. Agg is not interactive.
import matplotlib.pyplot as plt
import matplotlib.transforms
import matplotlib.offsetbox
import matplotlib.patches
import matplotlib.text
import matplotlib_shiftable_annotation

import numpy as np

import argparse
import json
import pandas as pd
from typing import Callable, Tuple, Union, List, Optional
import abc
import io
import os
import glob
import shutil
import logging
import logging.config
import yaml
import pprint
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
                                       "meanpartattrs": str,
                                       "bestind": np.int, "bestpartattrs": str,
                                       "bestpfrac": np.float, "bestpfracstddev": np.float,
                                       "medianind": np.int, "medianpartattrs": str,
                                       "medianpfrac": np.float, "medianpfracstddev": np.float,
                                       "worstind": np.int, "worstpartattrs": str,
                                       "worstpfrac": np.float, "worstpfracstddev": np.float,
                                       "candidatesdata": str}

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
                 input_rel_path: str = None,
                 output_to_file: bool = True,
                 output_to_stdout: bool = False,
                 log_generations: bool = True,
                 show_graph: bool = False,
                 signature_suffix: str = None,
                 optimization_input: dict = None) -> None:

        self.initial_mean = initial_mean
        self.initial_stddevs = initial_stddevs
        self.cma_options = cma_options if cma_options is not None else {}
        # Alternative (probably less safe): cma_options or {}
        self.rsa_parameters = rsa_parameters if rsa_parameters is not None else {}
        self.accuracy = accuracy
        self.parallel = parallel
        self.particle_attributes_parallel = particle_attributes_parallel
        self.output_to_file = output_to_file
        self.output_to_stdout = output_to_stdout
        self.log_generations = log_generations
        self.show_graph = show_graph
        self.optimization_input = optimization_input

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
        #  In run method, before self.CMAES assignement add printing optimization signature, and after optimization add
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
        self.parallel_simulations_number = min(self.parallel_threads_number, self.CMAES.popsize)
        self.omp_threads = self.parallel_threads_number // self.CMAES.popsize\
            if self.parallel and self.parallel_threads_number > self.CMAES.popsize else 1

        # Create file for logging generation data
        self.opt_data_filename = self.output_dir + "/optimization.dat" if self.log_generations else None
        if self.log_generations:
            # Create a file if it does not exist
            # with open(self.opt_data_filename, "w+"):
            #     pass
            with open(self.opt_data_filename, "w+") as opt_data_file:
                # Write header line
                opt_data_file.write("\t".join(self.optimization_data_columns) + "\n")

    # TODO Maybe define __setstate__ method to redirect output after unpickling
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
        unpicklable_attributes = ["stdout", "stderr", "logger"]
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
            # possible to specify file name after handler was instatiated)
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
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = StreamToLogger(logger=self.logger, log_level=logging.INFO)
        sys.stderr = StreamToLogger(logger=self.logger, log_level=logging.ERROR)

    def pickle(self) -> None:
        with open(self.output_dir + "/_" + self.__class__.__name__ + ".pkl", "wb") as pickle_file:
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
    @abc.abstractmethod
    def arg_to_particle_attributes(cls, arg: np.ndarray) -> str:
        """Function returning rsa3d program's parameter particleAttributes based on arg"""
        return ""

    @classmethod
    def arg_in_domain(cls, arg: np.ndarray) -> bool:
        """Function checking if arg belongs to the optimization domain"""
        return True

    @classmethod
    @abc.abstractmethod
    def draw_particle(cls, particle_attributes: str, scaling_factor: float, color: str)\
            -> matplotlib.offsetbox.DrawingArea:
        """
        Abstract class method drawing particle described by `particle_attributes` string attribute on
        matplotlib.offsetbox.DrawingArea and returning DrawingArea object.

        :param particle_attributes: Particle's particleAttributes rsa3d program's parameter string
        :param scaling_factor: Factor for scaling objects drawn on matplotlib.offsetbox.DrawingArea
        :param color: Particle's color specified by matplotlib's color string
        :return: matplotlib.offsetbox.DrawingArea object with drawn particle
        """
        pass

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
        simulation_output_dir = self.rsa_output_dir + "/" + simulation_labels.replace(",", "_")
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

    def rsa_simulation(self, candidate_num: int, arg: np.ndarray, particle_attributes: Optional[str] = None) -> int:
        """
        Function running simulations for particle shape specified by arg and waiting for rsa3d process.
        It assigns proper number of OpenMP threads to the rsa3d evaluation.

        :param candidate_num: Identification number of the candidate
        :param arg: Phenotype candidate's point
        :param particle_attributes: Particle attributes computed for arg - if not given, they are computed
        :return: RSA simulation's return code
        """

        sim_start_time = datetime.datetime.now()
        # Run a process with rsa3d program running simulation
        rsa_proc_arguments = self.rsa_proc_arguments[:]  # Copy the values of the template arguments
        if particle_attributes is None:
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
        simulation_output_dir = self.rsa_output_dir + "/" + simulation_labels.replace(",", "_")
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

        self.logger.info(msg="RSA simulation start: generation no. {}, candidate no. {}, simulation no. {}\n"
                             "Argument: {}\n"
                             "particleAttributes: {}".format(*simulation_labels.split(","),
                                                             pprint.pformat(arg),
                                                             particle_attributes))
        # TODO To be removed - for debugging
        # print("RSA simulation process call: {}\n".format(" ".join(rsa_proc_arguments)))
        with open(rsa_output_filename, "w+") as rsa_output_file:
            # Open a process with simulation
            with subprocess.Popen(rsa_proc_arguments,
                                  stdout=rsa_output_file,
                                  stderr=rsa_output_file,
                                  cwd=simulation_output_dir) as rsa_process:
                return_code = rsa_process.wait()
        sim_end_time = datetime.datetime.now()
        # Get collectors' number
        rsa_data_file_lines_count = subprocess.check_output(["wc", "-l",
                                                             glob.glob(simulation_output_dir + "/*.dat")[0]])
        collectors_num = int(rsa_data_file_lines_count.strip().split()[0])
        self.logger.info(msg="RSA simulation end: generation no. {}, candidate no. {}, simulation no. {}."
                             " Time: {}, collectors: {}, return code: {}".format(*simulation_labels.split(","),
                                                                                 str(sim_end_time - sim_start_time),
                                                                                 str(collectors_num),
                                                                                 str(return_code)))
        return return_code

    # TODO Maybe find a way to assign more threads to rsa processes when other processes in generation ended and there
    #  are unused threads - probably from the next collector on
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
        # It is said that multiprocessing module does not work with class instance method calls,
        # but in this case multiprocessing.pool.ThreadPool seems to work fine with the run_simulation method.
        with multiprocessing.pool.ThreadPool(processes=self.parallel_simulations_number) as pool:
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
        # return values.tolist()  # or list(values), because values.ndim == 1
        return list(values), return_codes

    def log_generation_data(self) -> None:
        func_data = pd.DataFrame(columns=["partattrs", "pfrac", "pfracstddev"])
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
                    func_data.loc[candidate_num] = [evaluation_data[4],
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
                           self.arg_to_particle_attributes(self.CMAES.mean),  # " ".join(map(str, self.CMAES.mean))
                           # TODO Maybe get also standard deviations from self.CMAES.logger or its output file
                           #  self.cmaes_output_dir + "/stddev.dat" - it should have data when this function is called
                           str(best_cand.name), best_cand.at["partattrs"],
                           str(best_cand.at["pfrac"]), str(best_cand.at["pfracstddev"]),
                           str(median_cand.name), median_cand.at["partattrs"],
                           str(median_cand.at["pfrac"]), str(median_cand.at["pfracstddev"]),
                           str(worst_cand.name), worst_cand.at["partattrs"],
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
        # TODO Maybe get also standard deviations from "stddev.dat" file
        opt_data_filename = _output_dir + "/" + signature + "/optimization.dat"
        # Data of the first generation (no. 0) is logged in xmean.dat file, but in some of other CMA-ES output files
        # it is not logged
        generations_mean_data = np.loadtxt(fname=mean_output_filename, comments=['%', '#'])
        with open(output_filename) as output_file, open(opt_data_filename, "w+") as opt_data_file:
            # Write header line
            opt_data_file.write("\t".join(cls.optimization_data_columns) + "\n")
            gen_num = 0
            func_data = pd.DataFrame(columns=["partattrs", "pfrac", "pfracstddev"])
            # TODO Maybe find more efficient or elegant solution

            def save_generation_data() -> None:
                func_data.sort_values(by="pfrac", ascending=False, inplace=True)
                mean_arg = generations_mean_data[gen_num, 5:]
                best_cand = func_data.iloc[0]
                median_cand = func_data.iloc[func_data.shape[0] // 2]
                worst_cand = func_data.iloc[-1]
                func_values_data = func_data.loc[:, ["pfrac", "pfracstddev"]]
                candidates = [val for ind, cand in func_values_data.iterrows()
                              for val in [ind, cand.at["pfrac"], cand.at["pfracstddev"]]]
                generation_data = [str(gen_num),
                                   cls.arg_to_particle_attributes(mean_arg),
                                   # " ".join(map(str, mean_arg))
                                   str(best_cand.name), best_cand.at["partattrs"],
                                   str(best_cand.at["pfrac"]), str(best_cand.at["pfracstddev"]),
                                   str(median_cand.name), median_cand.at["partattrs"],
                                   str(median_cand.at["pfrac"]), str(median_cand.at["pfracstddev"]),
                                   str(worst_cand.name), worst_cand.at["partattrs"],
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
                    func_data = pd.DataFrame(columns=["partattrs", "pfrac", "pfracstddev"])
                candidate_num = int(evaluation_labels[1])
                # If multiple lines in packing-fraction-vs-params.txt file correspond to the same candidate, the
                # values from the last such line will be used
                func_data.loc[candidate_num] = [evaluation_data[4],
                                                float(evaluation_data[1]),
                                                float(evaluation_data[2])]
            # Save last generation's data
            save_generation_data()

    @classmethod
    def plot_optimization_data(cls, signature: str, modulo: int = None) -> None:
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
        #                                    dtype={"names": ("generation_num", "meanpartattrs", "bestind", "bestpfrac"),
        #                                           # "formats": (np.int, str, np.int, np.float)}
        #                                           "formats": ("i4", "U", "i4", "f4")},  # Debugging
        #                                    delimiter="\t",  # Debugging
        #                                    skiprows=1,  # Skip header line
        #                                    usecols=(0, 1, 2, 3)  # tuple(range(len(cls.optimization_data_columns)))
        #                                    )

        # 2) Maybe use function fread from datatable package

        # 3) Solution with standard lines reading and filling pd.DataFrame:
        # optimization_data = pd.DataFrame(columns=list(columns))
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
        #     # If candidates' data is in separate columns, remove "candidatesdata" from cls.optimization_data_columns
        #     # and use restkey="candidatesdata" in csv.DictReader constructor.
        #     for record in opt_data_reader:
        #         pprint.pprint(record)

        # Loading optimization data using pd.read_table
        # Alternatively pass filepath_or_buffer=opt_data_filename to pd.read_table
        with open(opt_data_filename) as opt_data_file:
            optimization_data = pd.read_table(filepath_or_buffer=opt_data_file,
                                              index_col="generationnum")  # dtype=cls.optimization_data_columns
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

        # TODO Maybe use fig, ax = plt.subplots() and plot on axes
        # fig, ax = plt.subplots()
        # plt.rcParams["axes.autolimit_mode"] = "round_numbers"
        fig = plt.figure(num=args.signature, figsize=(10, 6.5))  # figsize is given in inches
        ax = plt.axes()
        plt.title("CMA-ES optimization of RSA mean packing fraction\nof fixed-radii polydisks")
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
                     fmt="bo-", capsize=2, label="Worst candidate's value")  # barsabove=True
        plt.errorbar(x=optimization_data.index, y=optimization_data["bestpfrac"],
                     yerr=optimization_data["bestpfracstddev"],
                     fmt="go-", capsize=2, label="Best candidate's value")  # barsabove=True
        plt.errorbar(x=optimization_data.index, y=optimization_data["medianpfrac"],
                     yerr=optimization_data["medianpfracstddev"],
                     fmt="ro-", capsize=2, label="Median candidate's value")  # barsabove=True
        plt.fill_between(optimization_data.index, optimization_data["worstpfrac"], optimization_data["bestpfrac"],
                         color="0.75")
        # plt.grid(axis="y")  # True, axis="y"
        plt.grid()
        leg = plt.legend()
        leg.set_draggable(True)
        # After dragging legend disappears, but reappears shifted after changing anything in the graph.
        # update="bbox" does not change this behaviour, but sets legend's position relative to figure, not axes, which
        # is bad.

        def particle_drawings_annotations(part_attrs_col: str, packing_frac_col: str, color: str, modulo: int,
                                          drawings_scale: float, drawings_offset: Tuple[float, float]) -> None:
            """
            Annotate packing fraction data series with draggable particle drawings

            :param part_attrs_col: Name of the column with particle attributes in optimization_data pd.DataFrame
            :param packing_frac_col: Name of the column with mean packing fractions in optimization_data pd.DataFrame
            :param color: Color of the particle drawings
            :param modulo: Annotate points in first, last and every modulo generation
            :param drawings_scale: Length of unitary segment in drawing (drawing's scale) given in fraction
                                   of x axis width
            :param drawings_offset: Tuple specifying annotation boxes offset in fraction of axes' width and height
            :return: None
            """
            # TODO Maybe scale drawings' paddings and arrows and boxes' frames relatively to graph's width and height,
            #  similarly as drawings are scaled
            # TODO Use an argument to specify for how many generations annotations are to be made
            #  or calculate it somehow.
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
                gen_nums.append(data_len - 1)
            for gen_num in gen_nums:
                part_attrs = optimization_data[part_attrs_col].at[gen_num]
                # Get particle drawing
                drawing_area = cls.draw_particle(particle_attributes=part_attrs,
                                                 scaling_factor=scaling_factor,
                                                 color=color)
                # Make annotation
                xy = (optimization_data.index[gen_num], optimization_data[packing_frac_col].at[gen_num])
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

                # drag_part_drawing = matplotlib.offsetbox.DraggableOffsetBox(ax, part_drawing)  # Not needed
                # Use matplotlib_shiftable_annotation.AnnotationBboxWithShifts for shiftability instead of draggability
                ab = matplotlib.offsetbox.AnnotationBbox(drawing_area,
                                                         xy=xy,
                                                         xybox=xy_box,
                                                         xycoords="data",
                                                         # boxcoords="axes fraction",
                                                         boxcoords="data",
                                                         pad=0.2,  # 0.4
                                                         fontsize=12,  # 12
                                                         # bboxprops={},
                                                         arrowprops=dict(arrowstyle="simple,"  # "->", "simple"
                                                                                    "head_length=0.2,"
                                                                                    "head_width=0.3,"  # 0.1, 0.5
                                                                                    "tail_width=0.01",  # 0.2
                                                                         facecolor="black",
                                                                         connectionstyle="arc3,"
                                                                                         "rad=0.3"))
                ax.add_artist(ab)
                # # AnnotationBbox subclasses matplotlib.text._AnnotationBase, so we can toggle draggability
                # # using the following method:
                ab.draggable()
                # ab.shiftable()
                # Maybe following is equivalent:
                # drag_ab = matplotlib.offsetbox.DraggableAnnotation(ab)

        data_len = len(optimization_data["bestpartattrs"])
        drawings_scale = 0.05
        if modulo is None:
            modulo = max(int(data_len * drawings_scale), 1)
        particle_drawings_annotations(part_attrs_col="worstpartattrs", packing_frac_col="worstpfrac", color="b",
                                      modulo=modulo, drawings_scale=drawings_scale, drawings_offset=(0.2, -0.1))  # -0.3
        # drawings_offset=(0.1, -0.1)
        particle_drawings_annotations(part_attrs_col="bestpartattrs", packing_frac_col="bestpfrac", color="g",
                                      modulo=modulo, drawings_scale=drawings_scale, drawings_offset=(0.2, 0.1))  # -0.1
        # drawings_offset = (0.1, 0.1)
        particle_drawings_annotations(part_attrs_col="medianpartattrs", packing_frac_col="medianpfrac", color="r",
                                      modulo=modulo, drawings_scale=drawings_scale, drawings_offset=(0.2, -0.1))  # -0.2
        # drawings_offset=(0.1, 0.)
        # ax.set_xlim(0, 1)
        # ax.set_ylim(0, 1)
        plt.show()

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

        self.logger.info(msg="")
        if self.CMAES.countiter == 0:
            self.logger.info(msg="Start of optimization")
            self.CMAES.logger.add()
        else:
            self.logger.info(msg="Start of resumed optimization")
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
            std_devs = self.CMAES.sigma * self.CMAES.sigma_vec.scaling * self.CMAES.sm.variances ** 0.5
            self.logger.info(msg=pprint.pformat(std_devs))
            self.logger.info(msg="Covariance matrix:")
            covariance_matrix = self.CMAES.sigma ** 2 * self.CMAES.sm.covariance_matrix
            for line in pprint.pformat(covariance_matrix).split("\n"):  # or .splitlines()
                self.logger.info(msg=line)
            self.logger.info(msg="Phenotype candidates:")
            for line in pprint.pformat(pheno_candidates).split("\n"):
                self.logger.info(msg=line)
            # values = self.evaluate_generation_parallel(pheno_candidates) if self.parallel\
            #     else self.evaluate_generation_serial(pheno_candidates)
            # TODO Maybe make evaluate_generation_* methods return values as np.ndarray
            if self.parallel:
                if self.particle_attributes_parallel:
                    values, return_codes = self.evaluate_generation_parallel_in_pool(pheno_candidates)
                else:
                    self.logger.info(msg="Computing candidates' particleAttributes parameters in series")
                    cand_particle_attributes = [self.arg_to_particle_attributes(arg) for arg in pheno_candidates]
                    self.logger.info(msg="Candidates' particleAttributes parameters:")
                    for line in pprint.pformat(cand_particle_attributes, width=200).split("\n"):
                        self.logger.info(msg=line)
                    values, return_codes = self.evaluate_generation_parallel_in_pool(pheno_candidates,
                                                                                     cand_particle_attributes)
                # TODO Implement computing it in parallel (probably using evaluate_generation_parallel_in_pool method,
                #  use also a function that for number of simulations and simulation number returns number
                #  of ompThreads)
                # TODO If computing in series, assign maximal number of OpenMP threads
                # TODO Maybe add an option to somehow tell the program to end optimization (e.g. kill -KILL an RSA
                #  process or send a signal to the main process)
                take_median = np.full(shape=len(pheno_candidates), fill_value=False)
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
                                # See https://docs.python.org/3/library/subprocess.html#subprocess.Popen.returncode
                                if sys.platform.startswith("linux"):
                                    signal_info = subprocess.check_output("kill -l " + str(-return_code), shell=True)
                                else:
                                    signal_info = subprocess.check_output(["kill", "-l", str(-return_code)])
                                signal_name = signal_info.decode().strip().upper()
                                warning_message += ", signal name: {}".format(signal_name)
                            # self.logger.debug(msg=signal_info)
                            # self.logger.debug(msg=signal_name)
                            self.logger.warning(msg=warning_message)
                            random_seed = "seed" in self.rsa_parameters if self.input_given\
                                else "seed" in self.all_rsa_parameters
                            # if return_code_name in ["", "TERM"] or (return_code_name == "USR1" and not random_seed):
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
                                return_codes[candidate_num] = self.rsa_simulation(candidate_num, candidate)
                            elif signal_name == "USR2":
                                # To set corresponding to RSA simulation phenotype candidate's value to the median
                                # of other candidates' values, kill simulation process with "kill -USR2 pid"
                                self.logger.warning(msg="Phenotype candidate's no. {} value will be set to the median"
                                                        " of other candidates' values".format(str(candidate_num)))
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
                                return_codes[candidate_num] = self.rsa_simulation(candidate_num, new_candidate)

                    with open(self.output_filename, "r") as rsa_output_file:
                        # TODO Maybe find more efficient or elegant solution
                        # TODO Maybe iterate through lines in file in reversed order - results of the current generation
                        #  should be at the end
                        for line in rsa_output_file:
                            evaluation_data = line.split("\t")
                            evaluation_labels = evaluation_data[0].split(",")
                            if int(evaluation_labels[0]) == self.CMAES.countiter:
                                read_candidate_num = int(evaluation_labels[1])
                                mean_packing_fraction = float(evaluation_data[1])
                                values[read_candidate_num] = -mean_packing_fraction
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
            # TODO Check, what happens in case when e.g. None is returned as candidate value, so (I guess)
            #  a reevaluation is conducted
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
            self.CMAES.tell(pheno_candidates, values)
            self.CMAES.logger.add()
            # Pickling of the object
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

        # TODO Add separate method for making graphs
        # TODO Maybe create another class for analyzing the results of optimization
        if self.show_graph:
            plot_cmaes_graph_in_background(self.CMAES.logger.name_prefix, self.signature)

        # if self.output_to_file:
        #     sys.stdout = stdout
        #     sys.stderr = stderr
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
    def draw_particle(cls, particle_attributes: str, scaling_factor: float, color: str)\
            -> matplotlib.offsetbox.DrawingArea:
        # Extract particle data
        # Scale polydisks so that they have unitary area
        part_data = np.array(particle_attributes.split(" ")[2:-1], dtype=np.float).reshape(-1, 3) \
                    / np.sqrt(np.float(particle_attributes.rpartition(" ")[2]))
        # Draw particle
        # Get polydisk's width and height
        x_min = np.min(part_data[:, 0] - part_data[:, 2])
        x_max = np.max(part_data[:, 0] + part_data[:, 2])
        y_min = np.min(part_data[:, 1] - part_data[:, 2])
        y_max = np.max(part_data[:, 1] + part_data[:, 2])
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
            disk_label = matplotlib.text.Text(x=scaling_factor * disk_args[0], y=scaling_factor * disk_args[1],
                                              text=str(disk_num),
                                              horizontalalignment="center",
                                              verticalalignment="center",
                                              fontsize=11)
            drawing_area.add_artist(disk_label)
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
    def arg_to_polydisk_attributes(cls, arg: np.ndarray) -> Tuple[str, np.ndarray]:
        """
        Function returning part of Polydisk's particleAttributes in a tuple, which first element is \"xy\" or \"rt\"
        string indicating type of coordinates and the second is a numpy ndarray with c01 c02 r0 c11 c12 r1 ... floats
        (disks' coordinates and radii)
        """
        arg_with_radii = np.insert(arg, np.arange(2, arg.size + 1, 2), 1.)
        return "xy", arg_with_radii


class ConstrFixedRadiiXYPolydiskRSACMAESOpt(PolydiskRSACMAESOpt):
    """
    Class for performing CMA-ES optimization of packing fraction of RSA packings built of unions of disks
    with unit radius. The last disk is placed at (0, 0), the last but one at (x, 0) and others are free.
    """

    default_rsa_parameters = dict(PolydiskRSACMAESOpt.default_rsa_parameters,  # super().default_rsa_parameters,
                                  **{  # "maxVoxels": "4000000",
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
    def arg_to_polydisk_attributes(cls, arg: np.ndarray) -> Tuple[str, np.ndarray]:
        """
        Function returning part of Polydisk's particleAttributes in a tuple, which first element is \"xy\" or \"rt\"
        string indicating type of coordinates and the second is a numpy ndarray with c01 c02 r0 c11 c12 r1 ... floats
        (disks' coordinates and radii)
        """
        arg_with_standard_disks_radii = np.insert(arg, np.arange(2, arg.size, 2), 1.)
        arg_with_all_disks = np.concatenate((arg_with_standard_disks_radii, np.array([0., 1., 0., 0., 1.])))
        return "xy", arg_with_all_disks


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

    opt_modes = {
        "optfixedradii": opt_fixed_radii,
        "optconstrfixedradii": opt_constr_fixed_radii
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
    modulo = int(args.modulo) if args.modulo is not None else None
    opt_class.plot_optimization_data(signature=args.signature, modulo=modulo)


def resume_optimization() -> None:
    if args.signature is None:
        raise TypeError("In resumeoptimization mode optimization signature has to be specified using -s argument")
    opt_class_name = args.signature.split("-")[5]
    # Get optimization class from current module.
    # If the class is not in current module, module's name has to be passed as sys.modules dictionary's key,
    # so such classes should put the module name to optimization signature.
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
    # Overwrite CMA-ES options, if the file argument was given. It should contain only the dictionary with CMAOptions
    # TODO Change this behaviour to accept not only CMA-ES options in input file but also optimization options that can
    #  be changed, and implement making these changes
    if args.file is not None:
        with open(_input_dir + "/" + args.file, "r") as opt_input_file:
            # TODO Maybe use configparser module or YAML format instead
            cmaes_options = json.load(opt_input_file)
        # After unpickling output is redirected to logger, so CMAEvolutionStrategy classes' errors and warnings
        # as e.g. "UserWarning: key popsize ignored (not recognized as versatile) ..." will be logged
        optimization.CMAES.opts.set(cmaes_options)
        # Generate used optimization input file in output directory
        resume_signature = datetime.datetime.now().isoformat(timespec="milliseconds")
        resume_signature += "-optimization-resume-input"
        resume_signature = resume_signature.replace(":", "-").replace(".", "_")
        opt_input_filename = optimization.output_dir + "/" + resume_signature + ".json"
        with open(opt_input_filename, "w+") as opt_input_file:
            json.dump(cmaes_options, opt_input_file, indent=2)
        optimization.logger.info(msg="Optimization resume input file: {}.json".format(resume_signature))
    # Run optimization
    optimization.run()


# TODO Add managing errors of the rsa3d program and forcing recalculation (or max n recalculations) or resampling of
#  the parameter point (maybe return None immediately after rsa_process.wait() in case of a failure) (old note)
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
#  methods for plotting and visualization of the saved data - partly DONE
#  then program mode with RSAParameters instantiation, calling run method and maybe other actions like plotting graphs
#  or making wolfram files etc. (Respective methods.) - partly DONE
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
    arg_parser.add_argument("-m", "--modulo", help="Annotate points with particles drawings in first, last and every"
                                                   "modulo generation. If not given,"
                                                   "modulo will be automatically adjusted.")
    args = arg_parser.parse_args()
    module_modes[args.mode]()
