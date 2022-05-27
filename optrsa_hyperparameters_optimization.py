"""
Optimization of hyperparameters of algorithm used for optimization
of packing fraction of two-dimensional Random Sequential Adsorption (RSA) packings.
"""
import sherpa

from typing import Optional
import datetime
import os
import ruamel.yaml
import shutil
from pathlib import Path
import logging
import logging.config
import sys
from optrsa import StreamToLogger, ExcludedLoggersFilter, StrFormatStyleMessageLogRecord, RSACMAESOptimization, \
    resume_optimization as optrsa_resume_optimization
from copy import deepcopy
import pickle
from mergedeep import merge
from module_arg_parser import ModuleArgumentParser


# Get absolute path to optrsa project directory
_proj_dir = os.path.dirname(__file__)
# Absolute paths to input and output directories
_input_dir = _proj_dir + "/input"
_output_dir = _proj_dir + "/output"
_sherpa_logging_config = _proj_dir + "/hyperparameters_optimization_logging.yaml"
libraries_info_logfile_excluded_loggers = ["optrsa_hyperparameters_optimization", "optrsa"]


logging.setLogRecordFactory(StrFormatStyleMessageLogRecord)


# TODO Use click instead, maybe together with click-config-file
mod_arg_parser = ModuleArgumentParser()

opt_input_file_arg_parser = mod_arg_parser.add_common_arg_parser("opt_input_file")
opt_input_file_arg_parser.add_argument("file", help="YAML optimization input file from ./input directory")

opt_signature_arg_parser = mod_arg_parser.add_common_arg_parser("opt_signature")
opt_signature_arg_parser.add_argument("signature", help="optimization signature - name of subdirectory of ./output")


yaml = ruamel.yaml.YAML()
yaml.indent(mapping=4, sequence=6, offset=4)


# class HyperparameterConfiguration(yaml.YAMLObject):
#     yaml_tag = "!param"
#
#     def __init__(self,
#                  range: Union[List[float], List[int], List[str]],
#                  type: str = "continuous",
#                  scale: str = "linear") -> None:
#         # TODO Add checking if the values of parameters are correct
#         self.range = range
#         self.type = type
#         self.scale = scale
#
#     @classmethod
#     def from_yaml(cls, loader, node):
#         configuration = loader.construct_mapping(node)
#         return cls(**configuration)


@ruamel.yaml.yaml_object(yaml)
class HyperparameterConfiguration:
    yaml_tag = "!param"

    def to_sherpa_parameter(self, name: str) -> sherpa.Parameter:
        if not hasattr(self, "range"):
            raise ValueError("Parameter {} does not have range specified".format(name))
        if not hasattr(self, "type") or self.type == "continuous":
            return sherpa.Continuous(name, self.range, getattr(self, "scale", "linear"))
        elif self.type == "discrete":
            return sherpa.Discrete(name, self.range, getattr(self, "scale", "linear"))
        elif self.type == "choice":
            return sherpa.Choice(name, self.range)
        elif self.type == "ordinal":
            return sherpa.Ordinal(name, self.range)


class RSACMAESOptHyperparametersOptimization:
    """
    Class for performing optimization of hyperparameters of CMA-ES optimization of RSA packing fraction
    and managing the output data
    """

    def __init__(self,
                 parameters: dict,
                 gpyopt_options: dict = None,
                 output_to_file: bool = True,
                 output_to_stdout: bool = False,
                 signature_suffix: str = None,
                 optimization_input: dict = None,
                 ) -> None:

        self.parameters = parameters
        self.gpyopt_options = gpyopt_options if gpyopt_options is not None else {}
        self.output_to_file = output_to_file
        self.output_to_stdout = output_to_stdout
        self.optimization_input = optimization_input

        # Set optimization signature
        self.signature = datetime.datetime.now().isoformat(timespec="milliseconds")  # Default timezone is right
        self.signature += "-" + type(self).__name__
        self.signature += ("-" + signature_suffix) if signature_suffix is not None else ""
        self.signature = self.signature.replace(":", "-").replace(".", "_")

        # Create output directory system
        self.output_dir = _output_dir + "/" + self.signature
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.optrsa_output_dir = self.output_dir + "/outoptrsa"
        if not os.path.exists(self.optrsa_output_dir):
            os.makedirs(self.optrsa_output_dir)
        # Maybe use shutil instead

        # Generate used optimization input file in output directory
        with open(self.output_dir + "/optimization-input.yaml", "w+") as opt_input_file:
            yaml.dump(self.optimization_input, opt_input_file)

        self.configure_logging()

        # TODO Maybe move these definitions to run method
        # Redirect output. To be moved to run method
        self.redirect_output_to_logger()

        # Settings for parallel computation
        # TODO Check, if using this parameter is necessary
        # self.parallel_threads_number = threads if threads is not None else os.cpu_count()  # * 2

        # Create parameter-sherpa parameters list and fixed parameters dictionary
        self.optimized_parameters = []
        self.fixed_parameters = {}

        def set_parameters(parameters_dict: dict, parent_keys: str = "") -> None:
            for key, value in parameters_dict.items():
                # This solution assumes that parameter keys do not contain "/" characters
                key_str = str(key)
                if "/" in key_str:
                    raise ValueError("Parameter name: {} contains forbidden \"/\" character".format(key_str))
                keys = parent_keys + "/" + key_str
                if isinstance(value, dict):
                    set_parameters(value, keys)
                elif isinstance(value, HyperparameterConfiguration):
                    self.optimized_parameters.append(value.to_sherpa_parameter(keys[1:]))
                else:
                    nested_fixed_parameters_dict = self.create_parent_parameter_keys(parent_keys[1:],
                                                                                     self.fixed_parameters)
                    nested_fixed_parameters_dict[key] = value
            if len(parameters_dict) == 0:
                self.create_parent_parameter_keys(parent_keys[1:], self.fixed_parameters)

        set_parameters(self.parameters)

        # Create parameter-sherpa optimization algorithm object
        self.algorithm = sherpa.algorithms.GPyOpt(max_concurrent=1,  # TODO Consider how to choose this parameter
                                                  verbosity=True,
                                                  **self.gpyopt_options)
        # Create parameter-sherpa study object
        self.study = sherpa.Study(parameters=self.optimized_parameters,
                                  algorithm=self.algorithm,
                                  lower_is_better=False,
                                  disable_dashboard=True,
                                  output_dir=self.output_dir)
        self.trial = None
        self.trial_output_dir = None

    def configure_logging(self) -> None:
        # Configure and set optimization state logger
        # Logs can be printed to a logfile or to the standard output. By default, logfile will contain log records of
        # severity level at least logging.INFO and standard output - at least logging.DEBUG.
        with open(_sherpa_logging_config) as config_file:
            logging_configuration = yaml.load(config_file)
        if self.output_to_file:
            # To set handlers' filename, modify configuration dictionary as below, try to set it to a variable in
            # configuration file, use logger's addHandler method or modify logger.handlers[0] (probably it is not
            # possible to specify file name after handler was instantiated)
            logging_configuration["handlers"]["hyperparameters_optimization_logfile"]["filename"] = self.output_dir \
                + "/optimization-output.log"
        else:
            logging_configuration["loggers"]["optrsa_hyperparameters_optimization.optimization"]["handlers"].pop(0)
            del logging_configuration["handlers"]["hyperparameters_optimization_logfile"]
        if self.output_to_stdout:
            logging_configuration["loggers"]["optrsa_hyperparameters_optimization.optimization"]["handlers"] \
                .append("debug_stdout")
        logging.config.dictConfig(logging_configuration)
        self.logger = logging.getLogger("optrsa_hyperparameters_optimization.optimization")
        # Configure root logger
        if len(logging.root.handlers) == 1:
            logging_root_handler = logging.root.handlers[0]
            if isinstance(logging_root_handler, logging.StreamHandler) and logging_root_handler.stream == sys.stderr:
                logging.root.removeHandler(logging_root_handler)
        if not logging.root.hasHandlers():
            formatter = self.logger.handlers[0].formatter
            warnings_logfile_handler = logging.FileHandler(filename=self.output_dir + "/warnings.log")
            warnings_logfile_handler.setLevel(logging.WARNING)
            warnings_logfile_handler.setFormatter(formatter)
            logging.root.addHandler(warnings_logfile_handler)
            libraries_info_logfile_handler = logging.FileHandler(filename=self.output_dir + "/libraries_info.log")
            libraries_info_logfile_handler.setLevel(logging.INFO)
            libraries_info_logfile_handler.setFormatter(formatter)
            libraries_info_logfile_handler.addFilter(ExcludedLoggersFilter(libraries_info_logfile_excluded_loggers))
            logging.root.addHandler(libraries_info_logfile_handler)

    def redirect_output_to_logger(self) -> None:
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        # If a decorator for redirecting output were used, a "with" statement could have been used
        sys.stdout = StreamToLogger(logger=self.logger, log_level=logging.INFO)
        sys.stderr = StreamToLogger(logger=self.logger, log_level=logging.ERROR)

    @staticmethod
    def create_parent_parameter_keys(parent_keys: str, parameters_dict: dict) -> dict:
        if parent_keys == "":
            return parameters_dict
        parent_keys_list = parent_keys.split("/")
        nested_parameters_dict = parameters_dict
        for parent_key in parent_keys_list:
            if parent_key not in nested_parameters_dict:
                nested_parameters_dict[parent_key] = {}
            nested_parameters_dict = nested_parameters_dict[parent_key]
        return nested_parameters_dict

    @staticmethod
    def create_optimization(optimization_input: dict):
        # TODO Try to annotate the return type - -> RSACMAESOptHyperparametersOptimization does not work.
        #  See https://stackoverflow.com/questions/44640479/mypy-annotation-for-classmethod-returning-instance
        opt_class_args = deepcopy(optimization_input)
        opt_class_args["optimization_input"] = optimization_input
        return RSACMAESOptHyperparametersOptimization(**opt_class_args)

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
        self.configure_logging()
        # Redirect output
        # TODO Maybe separate redirecting output from unpickling in order to be able to unpickle and use standard output
        self.redirect_output_to_logger()

    def pickle(self, name: Optional[str] = None) -> None:
        pickle_name = "" if name is None else "-" + name
        with open(self.output_dir + "/_" + self.__class__.__name__ + pickle_name + ".pkl", "wb") as pickle_file:
            pickle.dump(self, pickle_file)

    # TODO Annotate it correctly
    @classmethod
    def unpickle(cls, signature: str):
        with open(_output_dir + "/" + signature + "/_" + cls.__name__ + ".pkl", "rb") as pickle_file:
            return pickle.load(pickle_file)
        # Unpickling works outside this module provided that the class of pickled object and
        # HyperparameterConfiguration class are imported.

    def _evaluate_trial(self) -> float:
        trial_optrsa_parameters = {}
        for keys, value in self.trial.parameters.items():
            parent_keys, _, current_key = keys.rpartition("/")
            nested_parameters_dict = self.create_parent_parameter_keys(parent_keys, trial_optrsa_parameters)
            nested_parameters_dict[current_key] = value
        optrsa_parameters = dict(merge({},
                                       self.fixed_parameters,
                                       trial_optrsa_parameters,
                                       {"opt_class_args": {"output_dir": self.trial_output_dir.as_posix()}}))
        optrsa_optimization = RSACMAESOptimization.create_optimization(optrsa_parameters)
        optrsa_optimization.pickle()
        self.pickle()
        optrsa_optimization.run()
        # TODO Maybe deal with optrsa optimization errors
        objective = optrsa_optimization.get_result()[1]
        return objective

    def _resume_trial_evaluation(self) -> float:
        optrsa_opt_signature = str(os.listdir(self.trial_output_dir)[0])
        # Conversion to str is added because PyCharm assumes that os.listdir returns list of bytes, even though it
        # returns list of str
        optrsa_opt_resume_signature = self.signature + "/outoptrsa/{:03d}/".format(self.trial.id) \
            + optrsa_opt_signature
        # TODO Make it easier by using relative paths in optrsa module
        optrsa_resume_optimization(optrsa_opt_resume_signature)
        optrsa_opt_class_name = optrsa_opt_signature.split("-")[5]
        optrsa_opt_class = getattr(sys.modules["optrsa"], optrsa_opt_class_name)
        optrsa_optimization = optrsa_opt_class.unpickle(optrsa_opt_resume_signature)
        # TODO Maybe deal with optrsa optimization errors
        objective = optrsa_optimization.get_result()[1]
        return objective

    def _finalize_trial(self, objective: float) -> None:
        # TODO Maybe add observations after each generation (iteration), maybe supply some context
        self.study.add_observation(trial=self.trial,
                                   objective=objective)
        self.study.finalize(trial=self.trial)
        self.study.save()
        self.pickle()

    def run(self) -> None:
        """Method running optimization"""

        # TODO Parallelize optimizations in batches
        self.logger.info(msg="")
        if self.study.num_trials == 0:
            self.logger.info(msg="Start of optimization")
        else:
            self.logger.info(msg="Start of resumed optimization")
            trial_start_time = datetime.datetime.now()
            self.logger.info(msg="")
            self.logger.info(msg="Resuming trial number {}".format(self.trial.id))
            self.logger.info(msg="Trial parameters:")
            for parameter_name, parameter_value in self.trial.parameters.items():
                self.logger.info(msg="{}: {}".format(parameter_name, parameter_value))
            self.trial_output_dir = Path(self.optrsa_output_dir) / "{:03d}".format(self.trial.id)
            if self.trial_output_dir.exists():
                pickled_optimizations = list(self.trial_output_dir.glob("*/_*.pkl"))
                if len(pickled_optimizations) > 0:
                    objective = self._resume_trial_evaluation()
                else:
                    self.logger.warning(msg="Directory of trial number {} without optrsa pickle file found"
                                            " - this trial will be reevaluated".format(self.trial.id))
                    shutil.rmtree(self.trial_output_dir)
                    objective = self._evaluate_trial()
            else:
                objective = self._evaluate_trial()
            self._finalize_trial(objective)
            self.logger.info(msg="End of trial number {}".format(self.trial.id))
            self.logger.info(msg="Objective: {}".format(objective))
            trial_end_time = datetime.datetime.now()
            self.logger.info("Trial time: {}".format(str(trial_end_time - trial_start_time)))

        for self.trial in self.study:
            trial_start_time = datetime.datetime.now()
            self.logger.info(msg="")
            self.logger.info(msg="Trial number {}".format(self.trial.id))
            # self.logger.info(msg=("Trial number {}" if not optrsa_output_dir.exists() else "Resuming trial number {}")
            #                  .format(self.trial.id))
            self.logger.info(msg="Trial parameters:")
            for parameter_name, parameter_value in self.trial.parameters.items():
                self.logger.info(msg="{}: {}".format(parameter_name, parameter_value))
            self.trial_output_dir = Path(self.optrsa_output_dir) / "{:03d}".format(self.trial.id)
            objective = self._evaluate_trial()
            self._finalize_trial(objective)
            self.logger.info(msg="End of trial number {}".format(self.trial.id))
            self.logger.info(msg="Objective: {}".format(objective))
            trial_end_time = datetime.datetime.now()
            self.logger.info("Trial time: {}".format(str(trial_end_time - trial_start_time)))
        self.logger.info(msg="")
        self.logger.info(msg="End of optimization")
        self.logger.info(msg="")
        self.logger.info(msg="Best result:")
        for variable_name, variable_value in self.study.get_best_result().items():
            self.logger.info(msg="{}: {}".format(variable_name, variable_value))

        sys.stdout = self.stdout
        sys.stderr = self.stderr


def load_optimization_input(file: str) -> dict:
    with open(_input_dir + "/" + file, "r") as opt_input_file:
        return yaml.load(opt_input_file)


@mod_arg_parser.command(parsers=["opt_input_file"])
def optimize(file: str) -> None:
    """Run optimization"""
    optimization_input = load_optimization_input(file)
    optimization = RSACMAESOptHyperparametersOptimization.create_optimization(optimization_input)
    optimization.run()


@mod_arg_parser.argument("optimization_link",
                         help="file with optimization signature from ./input directory")
@mod_arg_parser.command("initializeopt", parsers=["opt_input_file"])
def initialize_optimization(file: str, optimization_link: str) -> None:
    """Initialize optimization"""
    optimization_input = load_optimization_input(file)
    optimization = RSACMAESOptHyperparametersOptimization.create_optimization(optimization_input)
    optimization.pickle()
    print("Optimization signature: {}".format(optimization.signature), file=optimization.stdout)
    with open(_input_dir + "/" + optimization_link, "w+") as opt_signature_file:
        opt_signature_file.write(optimization.signature)


@mod_arg_parser.argument("-f", "--file",
                         help="YAML resume-optimization input file from ./input directory")
@mod_arg_parser.command("resumeoptimization", parsers=["opt_signature"])
def resume_optimization(signature: str, file: Optional[str] = None) -> None:
    """Resume optimization"""
    optimization = RSACMAESOptHyperparametersOptimization.unpickle(signature)
    optimization.logger.info(msg="")
    optimization.logger.info(msg="")
    optimization.logger.info(msg="Resuming optimization")
    # Overwrite optimization options, if the file argument was given
    if file is not None:
        with open(_input_dir + "/" + file, "r") as opt_input_file:
            resume_input = yaml.load(opt_input_file)
        if "parameters" in resume_input:
            # Parameters are updated since the next trial, not the resumed one

            def set_resume_parameters(parameters_dict: dict, parent_keys: str = "") -> None:
                for key, value in parameters_dict.items():
                    # This solution assumes that parameter keys do not contain "/" characters
                    key_str = str(key)
                    if "/" in key_str:
                        raise ValueError("Parameter name: {} contains forbidden \"/\" character".format(key_str))
                    keys = parent_keys + "/" + key_str
                    if isinstance(value, dict):
                        set_resume_parameters(value, keys)
                    elif isinstance(value, HyperparameterConfiguration):
                        optimization.logger.warning(msg="Updating or setting new hyperparameters while resuming"
                                                        " optimization is not possible."
                                                        " {} parameter is ignored.".format(keys[1:]))
                    else:
                        if not any(optimized_parameter.name == keys[1:] for optimized_parameter
                                   in optimization.optimized_parameters):
                            nested_fixed_parameters_dict = optimization.create_parent_parameter_keys(
                                parent_keys[1:], optimization.fixed_parameters)
                            nested_fixed_parameters_dict[key] = value
                        else:
                            optimization.logger.warning(msg="Updating optimized parameters while resuming optimization"
                                                            " is not possible."
                                                            " {} parameter is ignored.".format(keys[1:]))
                if len(parameters_dict) == 0:
                    optimization.create_parent_parameter_keys(parent_keys[1:], optimization.fixed_parameters)

            set_resume_parameters(resume_input["parameters"])
        if "gpyopt_options" in resume_input:
            for option_name, option_value in resume_input["gpyopt_options"].items():
                setattr(optimization.study.algorithm, option_name, option_value)
            # optimization.gpyopt_options is not updated
        for attr in []:  # "nodes_number"
            if attr in resume_input:
                setattr(optimization, attr, resume_input[attr])
        # Generate used optimization input file in output directory
        resume_signature = datetime.datetime.now().isoformat(timespec="milliseconds").replace(":", "-") \
            .replace(".", "_")
        resume_signature += "-optimization-resume-trial-{}".format(optimization.study.num_trials)
        opt_input_filename = optimization.output_dir + "/" + resume_signature + "-input.yaml"
        with open(opt_input_filename, "w+") as opt_input_file:
            yaml.dump(resume_input, opt_input_file)
        optimization.logger.info(msg="Optimization resume input file: {}-input.yaml".format(resume_signature))
    # Run optimization
    optimization.run()


# TODO Maybe add feature of plotting optimization data
if __name__ == "__main__":
    mod_arg_parser()
