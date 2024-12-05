'''
Scrript for training agents based on a hermes file.
Example command for execution:
"python3 train.py -f hermes_files/rt_rareid_grp.hermes -n results  -ed /Users/nicola_mueller/Desktop/rare_tests/rareid"
'''
from rlmate.hermes import argparse, ExecutionFile, Execution
from pathlib import Path
import json

with open("all_args.json", "r") as f:
    all_args_dict = json.load(f)

def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        pass
    return False

def main():
    # parse command line arguments for script
    arg_parser = argparse.ArgumentParser(description='Train & evaluation argument configuration.')
    arg_parser.add_argument('--hermes_file', '-f' , help='The path to the hermes file to use for training')
    arg_parser.add_argument('--name', '-n', help='The name for hermes experiments')
    arg_parser.add_argument('--experiment_dir', '-ed',
        help='The directory where hermes experiments will be stored for this execution. Creates'
                + '"experiments" directory as subdirectory.')
    arg_parser.add_argument('--evaluate_only', '-eo', default=False, action='store_true',
        help='If set, hermes file will not be executed. Can be used to evaluate output of'
                + 'previous hermes file execution without rerunning it.')
    args = arg_parser.parse_args()

    args.comment = None
    args.debug = False

    exp_path = Path(args.experiment_dir)
    if not Path.is_dir(exp_path):
        raise Exception("The specified experiment root path {} is not a directory!\n".format(str(exp_path)))

    hermes_path = Path(args.hermes_file)
    if not Path.is_file(hermes_path):
        raise Exception("The specified hermes file ({}) does not exist!\n".format(str(hermes_path)))

    lines = None
    substitute_exp = False
    with open(args.hermes_file, "r") as f:
        lines = f.readlines()
        counter = 0
        for line in lines:
            if "-exp" in line:
                substitute_exp = True
                break
            else: 
                counter += 1

    if substitute_exp:
        exp_path = Path.joinpath(exp_path, "experiments")
        lines[counter] = "-exp {}\n".format(str(exp_path))

    with open(args.hermes_file, "w") as f:
        f.writelines(lines)

    execution_file = ExecutionFile(
            args.hermes_file, args.comment if args.comment else [], args.experiment_dir
        )

    execution = Execution(execution_file, args.name, args.debug)
    if not args.evaluate_only:
        execution.run()

if __name__ == "__main__":
    main()
