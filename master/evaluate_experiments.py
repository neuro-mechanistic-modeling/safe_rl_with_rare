"""
Script for evaluating trained agents.
Example command for execution:
"python3 evaluate_experiments.py -ed /Users/nicola_mueller/Desktop/rare_tests/rarepr/experiments -eb results"
"""

import argparse
from pathlib import Path
import os
import pandas as pd
import json
from concurrent.futures import ProcessPoolExecutor,  as_completed
import multiprocessing
import logging
import time
from rlmate.storage import Experiment
from evaluation_stage import EvaluationStage
from racetrackgym.environment import Environment as RacetrackEnvironment
from minigrid_environment import MiniGridDynamicObstacles, MiniGridDynamicObstaclesDoor, FlatFullyObsWrapper
from minigrid.wrappers import FullyObsWrapper
from argument_parser import RAREParser
import dqn, dqnpr, rareid, rarepr
import go_explore

logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.ERROR)

with open("all_args.json", "r") as f:
    all_args_dict = json.load(f)

agent_switcher = {
    "learn_dqn": dqn.DQNAgent,
    "learn_dqnpr": dqnpr.DQNPRAgent,
    "learn_rareid": rareid.RAREIDAgent,
    "learn_rarepr": rarepr.RAREPRAgent,
    "learn_go_explore": go_explore.GoExploreAgent,
}

def create_for_experiment(experiment, experiment_path, fn_agent, args):
    eval_stages = []

    initial_states = []
    if args.benchmark == "racetrack":
        env = RacetrackEnvironment(args)
        for starter in env.map.starters:
            starter = (starter[0], starter[1], 0, 0)
            initial_states.append(starter)

    elif args.benchmark == "minigrid":
        if args.map_name == "DynObs":
            env = FlatFullyObsWrapper(FullyObsWrapper(MiniGridDynamicObstacles(pos_r=args.positive_reward,
                                                                               neg_r=args.negative_reward,
                                                                               step_r=args.step_reward,
                                                                               n_obstacles=args.num_obstacles)))
        elif args.map_name == "DynObsDoor":
            env = FlatFullyObsWrapper(FullyObsWrapper(MiniGridDynamicObstaclesDoor(pos_r=args.positive_reward,
                                                                               neg_r=args.negative_reward,
                                                                               step_r=args.step_reward,
                                                                               n_obstacles=args.num_obstacles)))
        else:
            raise Exception("Invalid environment!")
        for starter in env.initial_states:
            x, y = starter
            initial_states.append((x, y, 0))

    for policy in experiment.list_pth_files():
        p_sign = policy.split("_")[-1][:-4]

        if p_sign == "checkpoint":
            continue

        args.gpu = False
        agent = fn_agent(env, args)
        agent.load(Path(experiment_path, policy))

        es = EvaluationStage(agent, args, env, p_sign, initial_states, args.es_grp)

        eval_stages.append(es)

    return eval_stages


def writer(q, results_path):
    file_exists = Path.is_file(results_path)

    while True:
        row = q.get()
        if row is None:
            break

        row.to_csv(results_path, mode='a', header=not file_exists)
        file_exists = True

    return True


def process_func_new(exp_dir, script_name, hermes_exp_name, batch_name, q):
    experiment_path = Path(exp_dir, script_name, batch_name, hermes_exp_name)

    agent = agent_switcher[script_name]
    parser = RAREParser

    experiment = Experiment.load(experiment_path)
    hermes_name = experiment_path.name
    tmp = hermes_name + " " + " ".join(experiment.exec_args)
    args = parser().parse(tmp)

    eval_stages = create_for_experiment(experiment, experiment_path, agent, args)
    for es in eval_stages:
        print("starting evaluation stage %s %s" % (str(es.args.hermes_name), str(es.sign)))
        es.args.es_initial_runs = 100
        es.args.es_kappa = 0.05
        start = time.time()
        if args.benchmark == "racetrack":
            es.eval(grps_eps=0.01, return_eps=1, both=True)

        elif args.benchmark == "minigrid":
            es.eval(grps_eps=0.01, return_eps=0.01, both=True)

        end = time.time()
        print("ended evaluation stage %s %s in %.2f seconds" % (
        str(es.args.hermes_name), str(es.sign), (end - start)))

        try:
            d = all_args_dict
            try:
                d.update(vars(es.args))
            except Exception as e:
                print(e)
            d["experiment_name"] = batch_name
            d["script"] = script_name
            d["sign"] = es.sign
            d["neural_network_weights"] = None  # breaks if you comment this
            d["grps_mean"] = es.get_mean_grps()
            d["grps_variance"] = es.get_mean_grps_variance()
            d["returns_mean"] = es.get_mean_returns()
            d["returns_variance"] = es.get_mean_returns_variance()
            d["json_file"] = ""
            d["betas"] = None
            print("\n")
            print("MEAN GRP " + " " + str(es.sign) + ": " + str(es.get_mean_grps()))
            print("\n")
            print("\n")
            print("MEAN RETURN " + " " + str(es.sign) + ": " + str(es.get_mean_returns()))
            print("\n")
        except Exception as e:
            print("Exception occurred when writing to dictionary!\n")
            print(str(e))

        current = pd.DataFrame(d, index=[0])
        q.put(current)

    return True


if __name__ == "__main__":
    multiprocessing.set_start_method('fork')

    # --- parse arguments for script ---
    arg_parser = argparse.ArgumentParser('argument configuration of evaluation of tuning experiments')
    arg_parser.add_argument(
        '-ed',
        '--exp_dir',
        help='The path to the directory with hermes experiment to be evaluated',
        default=''
    )
    arg_parser.add_argument(
        '-fn',
        '--file_name',
        help='The name of the CSV file which stores the results',
        default='res.csv'
    )
    arg_parser.add_argument(
        '-eb',
        '--exp_batch',
        default="results",
        help="The name of folder with the experiments to be evaluated. \
            Same as the name specified with -n for the executed hermes files.\
            If no name is specified, script evaluates every experiment batch!"
    )

    args = arg_parser.parse_args()

    # -- loop over experiments and create CSV file with evaluations ---
    experiment_dir = Path(args.exp_dir)
    assert (Path.is_dir(experiment_dir)), "Given experiment directory is not a directory!"
    results_path = Path.joinpath(experiment_dir, args.file_name)

    pool = ProcessPoolExecutor(max_workers=10)
    manager = multiprocessing.Manager()
    q = manager.Queue()
    writer_res = pool.submit(writer, q, results_path)
    jobs = []

    # loop through all agents in experiments
    for agent_name in os.listdir(experiment_dir):
        agent_dir = Path.joinpath(experiment_dir, agent_name)

        if not agent_dir.is_dir():
            continue

        # loop through the named experiment batches of an agent
        #for batch_name in os.listdir(agent_dir):
        for batch_name in [f for f in os.listdir(agent_dir) if not f.startswith('.')]:  # this is needed for macOS
            if args.exp_batch is not None:
                if args.exp_batch != batch_name:
                    continue

            batch_path = Path(agent_dir, batch_name)
            #for hermes_exp_name in os.listdir(batch_path):
            for hermes_exp_name in [f for f in os.listdir(batch_path) if not f.startswith('.')]:
                process_func_new(experiment_dir, agent_name, hermes_exp_name, batch_name, q)

                job = pool.submit(process_func_new, experiment_dir, agent_name, hermes_exp_name, batch_name, q)
                jobs.append(job)

    exceptions = 0

    num_jobs = len(jobs)
    finished = 0
    for job in as_completed(jobs):
        try:
            job.result()
        except Exception as e:
            exceptions += 1
            print("An exception has occured!\n")
            print(str(e))
        finished += 1
        print(f"\nFinished job [{finished}/{num_jobs}]\n")

    print(f"Finishing with {exceptions} exceptions")

    # message writer process that it can stop
    q.put(None)
    print("Waiting for writer to finish...")
    writer_res.result()
    print("Writer has finished")

    # free resources of process pool
    pool.shutdown(wait=False)
