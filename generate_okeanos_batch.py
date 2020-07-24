import optrsa
import argparse
import json
import numpy as np

account = "GB76-1"
partition = "okeanos"


def calculate_nodes_number(optimization_input: dict) -> int:
    max_nodes_number = optimization_input["max_nodes_number"] if "max_nodes_number" in optimization_input else None
    popsize = optimization_input["opt_class_args"]["cma_options"]["popsize"] \
        if "popsize" in optimization_input["opt_class_args"]["cma_options"] else None
    if popsize is not None:
        if max_nodes_number is not None:
            return min(max_nodes_number, popsize + 1)
        else:
            return popsize + 1
    else:
        points_number = optimization_input["opt_mode_args"]["vertices_num"] \
            if "polygon" in optimization_input["opt_mode"] \
            else optimization_input["opt_mode_args"]["disks_num"]
        dimensions_number = 2 * points_number
        if "constr" in optimization_input["opt_mode"]:
            dimensions_number -= 3
        if "fixedradii" not in optimization_input["opt_mode"]:
            dimensions_number += 1
        default_popsize = 4 + int(3 * np.log(dimensions_number))
        if max_nodes_number is not None:
            return min(max_nodes_number, default_popsize + 1)
        else:
            return default_popsize + 1


def main() -> None:
    module_description = "Module generating Okeanos SLURM batch scripts for optimizations with okeanos option"
    arg_parser = argparse.ArgumentParser(description=module_description)
    arg_parser.add_argument("input", help="optimization input file name")
    arg_parser.add_argument("file_name", help="batch file name")
    arg_parser.add_argument("job_name", help="job name")
    # arg_parser.add_argument("-N", "--max-nodes-number", help="maximum number of nodes to use")
    arg_parser.add_argument("-t", "--time", default="2-00:00:00", help="maximum job running time")
    arg_parser.add_argument("--mail-user", help="email address to send notifications")
    args = arg_parser.parse_args()
    # if args.file is None:
    #     raise TypeError("Batch filename has to be specified using -f argument")
    # if args.job_name is None:
    #     raise TypeError("Job name has to be specified using -J argument")
    # if args.input is None:
    #     raise TypeError("Optimization input filename has to be specified using -i argument")

    with open(optrsa._input_dir + "/" + args.input, "r") as opt_input_file:
        optimization_input = json.load(opt_input_file)
    nodes_number = calculate_nodes_number(optimization_input)
    output_file_name = optrsa._input_dir + "/" + args.file_name + ".out"
    mail_options = {}
    if args.mail_user is not None:
        mail_options = {
            "--mail-user": args.mail_user,
            "--mail-type": "ALL"
        }

    sbatch_options = {
        "-J": args.job_name,
        "-N": str(nodes_number),
        "--ntasks-per-node": "1",
        "-t": args.time,
        "--account": account,
        "--partition": partition,
        "--chdir": optrsa._proj_dir,
        "--output": output_file_name,
        **mail_options
    }
    commands = [
        "source optrsa-py-3-6-5-7-venv/bin/activate.csh",
        "srun python -m mpi4py.futures -m optrsa optimize -f {}".format(args.input)
    ]

    with open(optrsa._input_dir + "/" + args.file_name + ".batch", "a") as batch_file:
        batch_file.write("#!/bin/tcsh\n")
        batch_file.writelines(["#SBATCH {} {}\n".format(option_name, option_value)
                               for option_name, option_value in sbatch_options.items()])
        batch_file.write("\n")
        batch_file.writelines(map(lambda s: s + "\n", commands))


if __name__ == "__main__":
    main()
