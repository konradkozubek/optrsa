"""
Generation of SLURM batch script files for optrsa and optrsa_hyperparameters_optimization modules.
"""

import click
from pathlib import Path
from typing import Optional
from ruamel.yaml import YAML

sbatch_config_file = "optrsa_sbatch_config.yaml"
environment_file = "optrsa_environment.yaml"
sbatch_file = "optrsa_sbatch"
required_sbatch_options = [
    "job_name",
    "nodes",
    "time"
]
optional_sbatch_options = [
    "time_min"
]
sbatch_options_order = [
    "job_name",
    "nodes",
    "ntasks_per_node",
    "cpus_per_task",
    "time",
    "time_min",
    "account",
    "partition",
    "dependency",
    "requeue",
    "output",
    "open_mode",
    "mail_user",
    "mail_type"
]

yaml = YAML()


@click.command()
@click.argument("optimization_input",
                type=click.Path(exists=True, path_type=Path))
@click.option("-r", "--rsa", "optrsa_module", help="Use optrsa module.",
              flag_value="optrsa", default=True)
@click.option("-h", "--hyperparameters", "optrsa_module", help="Use optrsa_hyperparameters_optimization module.",
              flag_value="optrsa_hyperparameters_optimization")
@click.option("-d", "--script-directory", help="Path to directory where script file is to be created.",
              type=click.Path(exists=True, file_okay=False, path_type=Path), default=".")
@click.option("-J", "--job-name", help="Set --job-name SLURM option.",
              required=True)
@click.option("-N", "--nodes", help="Set --nodes SLURM option.",
              required=True)
@click.option("-t", "--time", help="Set --time SLURM option.",
              required=True)  # TODO Maybe use some time data format
@click.option("--time-min", help="Optionally set --time-min SLURM option.")  # TODO Maybe use some time data format
def main(optimization_input: Path,
         optrsa_module: str,
         script_directory: Path,
         job_name: str,
         nodes: str,
         time: str,
         time_min: Optional[str]) -> None:
    """
    Generate SLURM batch script file for optrsa or optrsa_hyperparameters_optimization module,
    using OPTIMIZATION_INPUT as optimization input file.
    """
    # Create script file
    with script_directory.joinpath("{}.sbatch".format(job_name)).open(mode="w") as script_file:
        script_file.write("#!/bin/bash\n")

        # Manage sbatch options and write them to script file
        # Load default sbatch options
        sbatch_options = yaml.load(Path(sbatch_config_file))
        local_variables = locals()
        # Set required sbatch options
        sbatch_options.update({option_name: local_variables[option_name] for option_name in required_sbatch_options})
        # Set optional sbatch options
        sbatch_options.update({option_name: local_variables[option_name] for option_name in optional_sbatch_options
                               if local_variables[option_name] is not None})
        # Set output option
        sbatch_options["output"] = script_directory.absolute().joinpath("%x.out").as_posix()
        # Order sbatch options
        sbatch_options_ordered = {option_name: sbatch_options.pop(option_name) for option_name in sbatch_options_order
                                  if option_name in sbatch_options}
        sbatch_options_ordered.update(sbatch_options)
        # Write sbatch options to script file
        script_file.writelines(["#SBATCH --{}{}\n".format(option_name.replace("_", "-"),
                                                          " {}".format(option_value) if option_value is not None
                                                          else "")
                                for option_name, option_value in sbatch_options_ordered.items()])
        script_file.write("\n")

        # Manage optrsa environment variables and write them to script file
        # Load default optrsa environment variables
        optrsa_variables = yaml.load(Path(environment_file))
        # Set required optrsa environment variables
        optrsa_variables.update({
            "optrsa_module": optrsa_module,
            "optimization_input": optimization_input.name  # TODO Change into optimization_input.as_posix()
        })
        # Write optrsa environment variables to script file
        script_file.writelines(["{}={}\n".format(variable_name.upper(), variable_value)
                                for variable_name, variable_value in optrsa_variables.items()])
        script_file.write("\n")

        script_file.write("source $OPTRSA_DIR/{}\n".format(sbatch_file))


if __name__ == "__main__":
    main()
