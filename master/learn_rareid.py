import sys
sys.path.append('../master')
from rareid import RAREIDAgent
from argument_parser import RAREParser
import logging
from minigrid.wrappers import FullyObsWrapper
from minigrid_environment import MiniGridDynamicObstacles, MiniGridDynamicObstaclesDoor, FlatFullyObsWrapper
from racetrackgym.environment import Environment

logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.DEBUG)


parser = RAREParser()
args = parser.parse()

if args.benchmark == "minigrid":
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
        raise Exception("Invalid MiniGrid environment!")

elif args.benchmark == "racetrack":
    env = Environment(args)
else:
    raise ValueError("Invalid benchmark")

a = RAREIDAgent(env, args)
a.train()
