"""
Slightly modified version of the evaluation stage from the original DSMC Evaluation Stages paper
"""
import copy
from racetrackgym.environment import Environment as RacetrackEnvironment
import logging
import numpy as np
import scipy.stats as st
import concurrent.futures
import time
import json
from scipy.stats import t
from rlmate.storage import Experiment
from dqn import Pseudo_agent as PA
from pathlib import Path
import os
from minigrid_environment import MiniGridDynamicObstacles, MiniGridDynamicObstaclesDoor, FlatFullyObsWrapper
from minigrid.wrappers import FullyObsWrapper
from argument_parser import RAREParser

logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.DEBUG)

def CH(kappa, eps):
    x = 1 / np.power(eps, 2)
    y = np.log(2 / (kappa))
    res = x * y

    return int(np.floor(res))


def S_square(ones, zeros):
    n = ones + zeros
    return ones / (n - 1) - np.power(ones, 2) / ((n - 1) * n)


def APMC(s2, kappa=0.05, eps=0.05):
    z = st.norm.ppf(1 - kappa / 2)
    return np.ceil(4 * z * s2 / np.power(eps, 2))


class EvaluationStage:
    def __init__(self, agent, args, env, sign, starters, grps_mode=True):
        self.grps_mode = grps_mode

        self.env = env
        self.agent = agent

        if self.agent.benchmark == "racetrack":
            self.x_len = env.map.height
            self.y_len = env.map.width
        elif self.agent.benchmark == "minigrid":
            self.x_len = env.width
            self.y_len = env.height
        else:
            raise ValueError("Invalid benchmark!")

        self.returns = [[[] for _ in range(self.y_len)] for _ in range(self.x_len)]
        self.losses = np.zeros((self.x_len, self.y_len))
        self.wins = np.zeros((self.x_len, self.y_len))
        self.tos = np.zeros((self.x_len, self.y_len))

        self.args = args
        logging.info("Initializing ES, min_prio %f" % self.args.es_minimal_prio)
        self.sign = sign

        self.starters = starters

        self.psi_min = args.psi_min
        self.psi_max = args.psi_max

    def get_coordinates(self, state):
        return state[0:2]

    def env_step(self, action):
        if self.agent.benchmark == "racetrack":
            return self.env.step(action)
        elif self.agent.benchmark == "minigrid":
            obs, reward, terminated, truncated, info = self.env.step(action)
            return reward, obs, terminated or truncated

    def get_grps(self, state=None, exhaustive=False):
        if state != None:
            x, y = state
            if exhaustive:
                return self.wins[x][y], self.losses[x][y], self.tos[x][y]
            else:
                return self.wins[x][y] / (
                    self.wins[x][y] + self.losses[x][y] + self.tos[x][y]
                )
        else:
            if exhaustive:
                return self.wins, self.losses, self.tos
            else:
                return self.wins / (self.wins + self.losses + self.tos)

    def get_mean_grps(self):
        grps_array = self.wins / (self.wins + self.losses + self.tos)
        return np.round(np.nanmean(grps_array), 3)

    def get_mean_grps_variance(self):
        grps_variance_array = self.get_grps_variance()
        return np.round(np.nanmean(grps_variance_array), 3)

    def get_mean_returns(self):
        returns = []
        for starter in self.starters:
            returns.append(self.get_returns(starter))
        return np.round(np.mean(returns), 3)

    def get_mean_returns_variance(self):
        returns_variance = []
        for starter in self.starters:
            returns_variance.append(self.get_returns_variance(starter))
        return np.round(np.mean(returns_variance), 3)

    def get_psi(self):
        if self.grps_mode:
            return self.get_psi_grp()
        else:
            return self.get_psi_return()

    def get_psi_grp(self):
        psi = self.get_mean_grps()

        if psi < self.psi_min:
            psi = self.psi_min
        elif psi > self.psi_max:
            psi = self.psi_max

        return psi

    def get_psi_return(self):
        # we first need to interpolate the evaluated expected returns to [0; 1]
        interpolated_returns = []
        for starter in self.starters:
            x, y = self.get_coordinates(starter)

            r = np.mean(self.returns[x][y])

            m = 1 / (self.args.positive_reward - self.args.negative_reward)

            b = 1 * (-self.args.negative_reward) / (self.args.positive_reward - self.args.negative_reward)

            interpolated = r * m + b

            interpolated_returns.append(interpolated)

        psi = np.mean(interpolated_returns)

        if psi < self.psi_min:
            psi = self.psi_min
        elif psi > self.psi_max:
            psi = self.psi_max

        return psi

    def get_returns(self, state=None):
        if state == None:
            res = np.zeros((self.x_len, self.y_len))
            for starter in self.starters:
                x, y = self.get_coordinates(starter)
                res[x][y] = np.mean(self.returns[x][y])
            return res
        else:
            x, y = self.get_coordinates(state)
            return np.mean(self.returns[x][y])

    def get_grps_dict(self):
        grps = self.get_grps()
        res = {}
        for starter in self.starters:
            x, y = self.get_coordinates(starter)
            if not self.env.map.terminal(x, y):
                res[(x, y)] = grps[x][y]
            return res

    def get_returns_dict(self):
        returns = self.get_returns()
        res = {}
        for starter in self.starters:
            x, y = self.get_coordinates(starter)
            if not self.env.terminal(x, y):
                res[(x, y)] = returns[x][y]
            return res

    def _interpolate_returns(self):
        returns = self.get_returns()
        b = (
            np.ones((self.x_len, self.y_len))
            * (-self.args.negative_reward)
            / (self.args.positive_reward - self.args.negative_reward)
        )
        m = np.ones((self.x_len, self.y_len)) / (
            self.args.positive_reward - self.args.negative_reward
        )

        return returns * m + b

    def get_grps_variance(self, state=None):
        if state == None:
            return S_square(self.wins, self.losses + self.tos)
        else:
            x, y = self.get_coordinates(state)
            return S_square(self.wins[x][y], self.losses[x][y] + self.tos[x][y])

    def get_returns_variance(self, state=None):
        if state == None:
            return np.var(self.returns, axis=2, ddof=1)
        else:
            x, y = self.get_coordinates(state)
            return np.var(self.returns[x][y], ddof=1)

    def evaluation_regrets(self, best_evals, dictionary_mode=False):
        if self.grps_mode:
            curr_evals = self.get_grps()
        else:
            curr_evals = self._interpolate_returns()

        if dictionary_mode:
            priorities = {}
        else:
            priorities = np.zeros((self.x_len, self.y_len))

        new_best_evals = copy.deepcopy(best_evals)

        for starter in self.starters:
            x, y = self.get_coordinates(starter)

            assert best_evals[x][y] >= -1 and best_evals[x][y] <= 1

            if curr_evals[x][y] < 0.0:
                curr_evals[x][y] = 0.0
            elif curr_evals[x][y] > 1.0:
                curr_evals[x][y] = 1.0

            if best_evals[x][y] == -1:
                regret = 1 - curr_evals[x][y]
            else:
                regret = best_evals[x][y] - curr_evals[x][y]

            min_regret = -1
            max_regret = 1
            b = -min_regret / (max_regret - min_regret)
            m = 1 / (max_regret - min_regret)

            interpolated_regret = regret * m + b

            tmp = np.power(
                (interpolated_regret + self.args.es_minimal_prio), self.args.es_alpha
            )

            if dictionary_mode:
                priorities[(x, y)] = tmp
            else:
                priorities[x][y] = tmp

            if curr_evals[x][y] > best_evals[x][y]:
                new_best_evals[x][y] = curr_evals[x][y]

        return priorities, new_best_evals

    def get_regrets(self, best_evals):
        if self.grps_mode:
            curr_evals = self.get_grps()
        else:
            curr_evals = self._interpolate_returns()

        regrets = np.zeros((self.x_len, self.y_len))

        for starter in self.starters:
            x, y = self.get_coordinates(starter)

            if curr_evals[x][y] < 0.0:
                curr_evals[x][y] = 0.0
            elif curr_evals[x][y] > 1.0:
                curr_evals[x][y] = 1.0

            assert best_evals[x][y] >= -1 and best_evals[x][y] <= 1

            if best_evals[x][y] == -1:
                regret = 1 - curr_evals[x][y]
            else:
                regret = best_evals[x][y] - curr_evals[x][y]

            min_regret = -1
            max_regret = 1
            b = -min_regret / (max_regret - min_regret)
            m = 1 / (max_regret - min_regret)

            interpolated_regret = regret * m + b

            regrets[x][y] = interpolated_regret

        return regrets

    def priorities(self, dictionary_mode=False):
        logging.info("Getting Priorities. Min Prio %f, alpha %f" % (self.args.es_minimal_prio, self.args.es_alpha))
        if self.grps_mode:
            p = self.get_grps()
        else:
            p = self._interpolate_returns()
        if dictionary_mode:
            res = {}
        else:
            res = np.zeros((self.x_len, self.y_len))
        for starter in self.starters:
            x, y = self.get_coordinates(starter)

            tmp = np.power(
                (1 - p[x][y] + self.args.es_minimal_prio), self.args.es_alpha
            )
            if dictionary_mode:
                res[(x, y)] = tmp
            else:
                res[x][y] = tmp
        return res

    def sim_len(self, state):
        x, y = self.get_coordinates(state)
        return len(self.returns[x][y])

    def _update(self, x, y, ret, reward):
        if reward > 0:
            self.wins[x][y] += 1
        elif reward < 0:
            self.losses[x][y] += 1
        else:
            self.tos[x][y] += 1

        self.returns[x][y].append(ret)

    def _simulate_runs(self, init_state, n_episodes, env=None):
        if self.agent.benchmark == "racetrack":
            self._simulate_runs_racetrack(init_state, n_episodes, env)
        elif self.agent.benchmark == "minigrid":
            self._simulate_runs_minigrid(init_state, n_episodes, env)

    def _simulate_runs_racetrack(self, init_state, n_episodes, env=None):
        if env == None:
            env = self.env
        x, y, a, b = init_state
        for _ in range(n_episodes):
            env.reset_to_state((x, y), (a, b))  # reset to state with velocity
            state = env.get_state()

            assert [x, y, a, b] == state[0:4]

            ret = 0
            for t in range(self.args.length_episodes):
                action = self.agent.act(state)
                r, state, done = env.step(action)
                ret += r * np.power(self.args.gamma, t)
                if done:
                    break
            self._update(x, y, ret, r)

    def _simulate_runs_minigrid(self, init_state, n_episodes, env=None):
        if env == None:
            env = self.env

        for _ in range(n_episodes):
            # if we restart in an initial state we only care about position and direction
            if len(init_state) == 3:
                x, y, d = init_state
                # set starting state
                env.set_start(x=x, y=y, d=d)
                # reset grid
                state, info = env.reset()
            # when restarting in an archived state we want to restore the exact state
            elif len(init_state) == 4:
                state, info = env.reset_to_state(init_state)
            else:
                raise ValueError("Invalid initial state!")

            ret = 0
            for t in range(self.args.length_episodes):
                # agent returns actions as integers
                action_num = self.agent.act(state)
                # minigrid requires action objects
                action = env.num_to_action(action_num)
                # take step in environment
                state, r, terminated, truncated, info = env.step(action)

                ret += r * np.power(self.args.gamma, t)
                if terminated or truncated:
                    break

            x, y = init_state[0:2]
            self._update(x, y, ret, r)

    def _construct_confidence_interval_length(self, state):
        if self.grps_mode:
            var = self.get_grps_variance(state)
        else:
            var = self.get_returns_variance(state)

        std = np.sqrt(var)
        n = self.sim_len(state)
        t_stat = t(df=n - 1).ppf((self.args.es_kappa / 2, 1 - self.args.es_kappa / 2))[
            -1
        ]

        return 2 * (t_stat * std / np.sqrt(n))

    def _construct_both_confidence_interval_lengths(self, state):
        var_grps = self.get_grps_variance(state=state)
        var_r = self.get_returns_variance(state=state)
        std_grps = np.sqrt(var_grps)
        std_r = np.sqrt(var_r)
        n = self.sim_len(state=state)
        t_stat = t(df=n - 1).ppf((self.args.es_kappa / 2, 1 - self.args.es_kappa / 2))[
            -1
        ]

        return 2 * (t_stat * std_r / np.sqrt(n)), 2 * (t_stat * std_grps / np.sqrt(n))

    def _eval_state_grp(self, state, env):
        ch_bound = CH(self.args.es_kappa, self.args.es_epsilon)
        interval_length = float("inf")
        made_runs = 0
        apmc_bound = float("inf")
        while (
            made_runs < apmc_bound
            and made_runs < ch_bound
            and interval_length > 2 * self.args.es_epsilon
        ):
            self._simulate_runs(state, self.args.es_initial_runs, env)
            made_runs += self.args.es_initial_runs
            apmc_bound = APMC(
                self.get_grps_variance(state), self.args.es_kappa, self.args.es_epsilon
            )
            interval_length = self._construct_confidence_interval_length(state)

    def _eval_state_returns(self, state, env):
        interval_length = float("inf")
        while interval_length > 2 * self.args.es_epsilon:
            self._simulate_runs(state, self.args.es_initial_runs, env)
            interval_length = self._construct_confidence_interval_length(state)

    def _eval_state(self, state, env=None):
        start = time.time()

        if env == None:
            env = self.env
        if self.grps_mode:
            self._eval_state_grp(state, env)
        else:
            self._eval_state_returns(state, env)

        end = time.time()
        print("Finished evaluation of state %s in %.2f sec using %d episodes"%(str(state), (end-start), self.sim_len(state)))

        return True

    def _eval_both_state(self, state, env, grps_eps, return_eps):
        start = time.time()
        ch_bound = CH(self.args.es_kappa, grps_eps)
        made_runs = 0

        if env == None:
            env = self.env

        while True:
            self._simulate_runs(state, self.args.es_initial_runs, env)

            made_runs += self.args.es_initial_runs
            apmc_bound = APMC(self.get_grps_variance(state=state), self.args.es_kappa, grps_eps)

            (
                interval_length_r,
                interval_length_grp,
            ) = self._construct_both_confidence_interval_lengths(state)

            return_criterion = interval_length_r < 2 * return_eps
            grps_criterion = (
                (made_runs > apmc_bound)
                or (made_runs > ch_bound)
                or (interval_length_grp < 2 * grps_eps)
            )

            if return_criterion and grps_criterion:
                break

        end = time.time()
        print("Finished evaluation of state %s in %.2f sec using %d episodes"%(str(state), (end-start), self.sim_len(state)))



    def eval(self, both = False, grps_eps = 0.1, return_eps = 4):
        ts = np.zeros((self.x_len, self.y_len)).tolist()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.args.es_num_threads
        ) as executor:
            for starter in self.starters:
                x, y = self.get_coordinates(starter)
                if both:
                    t = executor.submit(self._eval_both_state, starter,  self.env.clone(), grps_eps, return_eps)
                else:
                    t = executor.submit(
                        self._eval_state, starter, self.env.clone()
                    )
                ts[x][y] = t

            for starter in self.starters:
                x, y = self.get_coordinates(starter)
                ts[x][y].result()

    def save_results(self, savePath = None):
        if savePath:
            savePath = Path.joinpath(Path(savePath), "json_files")
            if not Path.is_dir(savePath):
                os.makedirs(savePath, exist_ok=True)
            save_file = Path.joinpath(savePath, self.args.hermes_name + "_" + self.sign + "_es.json")
        else:
            save_file = self.args.hermes_name + "_" + self.sign + "_es.json"
        with open(save_file, "w") as f:
            f.write(json.dumps(self.returns) + "\n")
            f.write(json.dumps(self.wins.tolist()) + "\n")
            f.write(json.dumps(self.losses.tolist()) + "\n")
            f.write(json.dumps(self.tos.tolist()))
        return save_file

    def load_results(self, path):
        with open(path, "r") as f:
            self.returns = json.loads(f.readline())
            self.wins = np.array(json.loads(f.readline()))
            self.losses = np.array(json.loads(f.readline()))
            self.tos = np.array(json.loads(f.readline()))

    @classmethod
    def load(cls, path):
        experiment_path = path.parent
        experiment = Experiment.load(experiment_path)
        hermes_name = experiment_path.name
        tmp = [hermes_name] + " ".join(experiment.exec_args).split()
        args = RAREParser().parse(tmp)

        if args.benchmark == "racetrack":
            env = RacetrackEnvironment(args)

        elif args.benchmark == "minigrid":
            if args.environment == "DynObs":
                env = FlatFullyObsWrapper(FullyObsWrapper(MiniGridDynamicObstacles(pos_r=args.positive_reward,
                                                                                   neg_r=args.negative_reward,
                                                                                   step_r=args.step_reward,
                                                                                   n_obstacles=args.num_obstacles)))
            elif args.environment == "DynObsDoor":
                env = FlatFullyObsWrapper(FullyObsWrapper(MiniGridDynamicObstaclesDoor(pos_r=args.positive_reward,
                                                                                       neg_r=args.negative_reward,
                                                                                       step_r=args.step_reward,
                                                                                       n_obstacles=args.num_obstacles)))
            else:
                raise Exception("Invalid environment!")

        net = experiment.load_network_class()("cpu", weights=args.neural_network_weights)
        sign = path.name.split("_")[-1][:-4]
        found_policy = False
        for policy in experiment.list_pth_files():
            p_sign = policy.split("_")[-1][:-4]
            if sign == p_sign:
                experiment.load_state_dict(policy, net)
                found_policy = True
                break
        if not found_policy:
            logging.warning(
                "Failed to load policy. Taking a randomly initialised network instead"
            )

        agent = PA(net)

        es = cls(agent, args, env, sign, args.es_grp)
        es.load_results(path)

        return es
