"""
An implementation of the original Go-Explore algorithm that restores states like RARE and learns according to DQN.
"""
import random
from collections import deque
import numpy as np
from rareid import RAREIDAgent
import json
import racetrackgym.graphic as g

# archive that stores for each cell of the environment's grid the states from the trajectories with the highest returns
class Archive:
    def __init__(self, args, env, env_width, env_height, initial_states):
        self.env = env
        self.args = args

        # depending on the benchmark we need different data structures for the archive's cells since the states are
        # represented differently
        if self.args.benchmark == "racetrack":
            self.cells = np.ones((env_width, env_height, 4), dtype=np.int32) * -1
            self.spawnable_positions = self.env.map.spawnable_positions
            for initial_state in initial_states:
                x, y = initial_state[0:2]
                self.cells[x][y] = np.array([(x, y, 0, 0)])

        elif self.args.benchmark == "minigrid":
            self.cells = [[[] for _ in range(env_height)] for _ in range(env_width)]
            self.spawnable_positions = self.env.spawnable_positions
            for initial_state in initial_states:
                x, y = initial_state[0:2]
                self.cells[x][y] = (x, y, 0)

        # tracks how often a cell has been visited
        self.seen = np.zeros((env_width, env_height))
        # tracks the highest returns achieved from each cell
        self.returns = np.ones((env_width, env_height), dtype=np.float32) * args.negative_reward
        self.trajectories = [[[] for _ in range(env_height)] for _ in range(env_width)]

        self.state_ids = np.arange(len(self.spawnable_positions))
        self.sampling_probabilities = []
        self.compute_sampling_probabilities()

    # adds a state and the following trajectory to the archive if the resulting return is higher than that of any previously stored state
    def add_state(self, state, ret, trajectory):
        x, y = state[0:2]

        # avoid overwriting the initial states in Racetrack
        if self.args.benchmark == "racetrack":
            #if (x, y) in self.env.map.starters:
            #    return
            state = np.array([state])

        self.seen[x][y] += 1
        if ret > self.returns[x][y]:
            self.cells[x][y] = state
            self.returns[x][y] = ret
            # print("Trajectory: ", trajectory)
            self.trajectories[x][y] = trajectory

    # update sampling probabilities for cells from each state according to how often each cell has been visited
    def compute_sampling_probabilities(self):
        sampling_priorities = np.zeros(len(self.spawnable_positions))
        for i, (x, y) in enumerate(self.spawnable_positions):
            sampling_priorities[i] = 1 / np.sqrt(self.seen[x][y] + 1)

        self.sampling_probabilities = sampling_priorities / sampling_priorities.sum()

    # samples state from archive
    def sample_state(self):
        for i in range(100000):
            state_id = np.random.choice(self.state_ids, p=self.sampling_probabilities)
            x, y = self.spawnable_positions[state_id]
            sampled_state = self.cells[x][y]

            if self.args.benchmark == "racetrack":
                if sampled_state[0] != -1:
                    assert not self.env.map.terminal(x, y)
                    return sampled_state

            elif self.args.benchmark == "minigrid":
                if len(self.cells[x][y]) != 0:
                    assert not self.env.terminal(x, y)
                    return sampled_state

    # sample from the archive the trajectories with the highest returns
    def get_best_trajectories(self, trajectory_n):
        return_sampling_priorities = np.zeros(len(self.spawnable_positions))
        for i, (x, y) in enumerate(self.spawnable_positions):
            ret = self.returns[x][y]
            interpolated_return = (ret - self.args.negative_reward) / (self.args.positive_reward - self.args.negative_reward)
            assert 0 <= interpolated_return <= 1

            # ensures that all trajectories have a non-zero chance to be sampled
            return_sampling_priorities[i] = interpolated_return + 0.1

        return_sampling_probabilities = return_sampling_priorities / return_sampling_priorities.sum()

        best_trajectories = []
        while len(best_trajectories) < trajectory_n:
            for i in range(10000):
                state_id = np.random.choice(self.state_ids, p=return_sampling_probabilities)
                x, y = self.spawnable_positions[state_id]
                sampled_trajectory = self.trajectories[x][y]

                if self.args.benchmark == "racetrack":
                    if self.cells[x][y][0] != -1:
                        assert not self.env.map.terminal(x, y)
                        best_trajectories.append(sampled_trajectory)
                        break
                elif self.args.benchmark == "minigrid":
                    if len(self.cells[x][y]) != 0:
                        assert not self.env.terminal(x, y)
                        best_trajectories.append(sampled_trajectory)
                        break

        if self.args.benchmark == "minigrid":
            # In minigrid, always include one trajectory where the agent needs to start in an initial state
            initial_idx = np.arange(len(self.env.initial_states))
            p = np.ones(len(self.env.initial_states))
            p = p / p.sum()
            initial_id = np.random.choice(initial_idx, p=p)
            x, y = self.env.initial_states[initial_id]
            initial_state = (x, y, 0)
            best_trajectories.append([[initial_state, -1, -1]])

        return best_trajectories


class GoExploreAgent(RAREIDAgent):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.args = args
        self.archive = Archive(self.args, self.env, self.env_width, self.env_height, self.initial_states)

        self.exploration_n = 100
        self.robustify_max_attempts = 10
        if self.benchmark == "racetrack":
            self.trajectory_n = 10
        elif self.benchmark == "minigrid":
            self.trajectory_n = 9

    # alternate between go, explore, and robustify
    def train(self):
        self.episodes_counter = 0
        while self.episodes_counter <= self.n_episodes:
            for i in range(self.exploration_n):
                state = self.go()
                self.explore(state)

            self.archive.compute_sampling_probabilities()

            self.robustify()

        self.save_network("end")
        self.save_counters()

    # sample state from archive and restore to it
    def go(self):
        starter = self.archive.sample_state()
        state = self.restore_state(starter)

        if self.args.benchmark == "racetrack":
            #print("GO: " + str((state[0:4])))
            x, y = state[0:2]
        elif self.args.benchmark == "minigrid":
            #print("GO: " + str((state[-3:])))
            x, y = self.env.agent_start_pos
        self.counter_starting_states[x][y] += 1

        return state

    # sample random actions to explore environment and store states with high returns in archive
    def explore(self, state):
        #if self.args.benchmark == "racetrack":
        #    print("EXPLORE FROM: " + str((state[0:4])))
        #elif self.args.benchmark == "minigrid":
        #    print("EXPLORE FROM: " + str((state[-3:])))

        # run episode using random exploration and store trajectory
        trajectory = []
        for t in range(self.l_episodes):
            transition = []
            if self.args.benchmark == "racetrack":
                transition.append(state[0:4])
            elif self.args.benchmark == "minigrid":
                transition.append(self.env.compress_state())

            action = random.choice(range(self.action_space_dim))
            reward, next_state, done = self.env_step(action)

            transition.append(reward)
            transition.append(0)
            assert len(transition) == 3
            trajectory.append(transition)
            state = next_state

            if done:
                break

        # compute returns achieved from each state
        current_return = 0
        for transition in reversed(trajectory):
            reward = transition[1]
            current_return = reward + self.gamma * current_return

            assert transition[2] == 0
            transition[2] = current_return

        # check for each state whether it should be archived
        for i in range(len(trajectory)):
            current_trajectory = trajectory[i:]
            current_transition = current_trajectory[0]

            state = current_transition[0]
            ret = current_transition[2]

            self.archive.add_state(state, ret, current_trajectory)

    # robustify on best trajectories from archive according to DQN
    def robustify(self):
        trajectories = self.archive.get_best_trajectories(self.trajectory_n)

        for trajectory in trajectories:
            #print("TRAJECTORY: " + str(trajectory))
            for transition in reversed(trajectory):
                current_state = transition[0]
                current_ret = transition[2]

                #if self.args.benchmark == "racetrack":
                #    print("ROBUSTIFY from " + str(current_state[0:4]) + " until return is " + str(current_ret))
                #elif self.args.benchmark == "minigrid":
                #    print("ROBUSTIFY from " + str(current_state[-3:]) + " until return is " + str(current_ret))

                # initialize arrays and values
                self.means = []
                self.scores_window = deque(maxlen=100)

                attempts = 0

                # run episodes until attempts have been exhausted or archived return has been achieved
                while True:
                    # reset state and score
                    state = self.restore_state(current_state)
                    score = 0
                    for t in range(self.l_episodes):
                        action = self.act(state, self.eps)

                        # send the action to the environment and observe
                        reward, next_state, done = self.env_step(action)

                        # add the sample to the buffer
                        self.update_buffer(state, action, reward, next_state, done)

                        # if the update counter meets the requirement of UPDATE_EVERY,
                        # sample and start the learning process
                        # increment update counter
                        self.update_counter = (self.update_counter + 1) % self.dqn_args.update_every
                        if self.update_counter == 0:
                            if (len(self.buffer)) > self.dqn_args.batch_size:
                                samples = self.buffer.sample()
                                self.learn(samples, self.gamma)

                        state = next_state
                        score += reward * np.power(self.gamma, t)
                        if done:
                            break

                    # increase episode counter
                    self.episodes_counter += 1

                    # store achieved score
                    self.scores_window.append(score)

                    # update exploration epsilon
                    self.eps = max(self.eps_end, self.eps_decay * self.eps)

                    self.logging(score)

                    if self.args.print_heatmaps:
                        if self.args.policy_extraction_frequency != 0 and (
                                self.episodes_counter % self.args.policy_extraction_frequency
                        ) == (-1 % self.args.policy_extraction_frequency):
                            # print how often each cell has been seen
                            g.print_heatmap(
                                self.archive.seen,
                                print_path=self.args.hermes_name + "_seen-heatmap_" + str(self.episodes_counter) + ".png",
                                show=False,
                                bounds=(np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]) * 1000).tolist()
                            )
                            # print returns achieved for each cell
                            g.print_heatmap(
                                self.archive.returns,
                                print_path=self.args.hermes_name + "_returns-heatmap_" + str(self.episodes_counter) + ".png",
                                show=False,
                                bounds=[-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
                            )

                    # break if all attempts have been used
                    attempts += 1
                    if attempts >= self.robustify_max_attempts:
                        #print("max attempts exhausted")
                        break

                    # break if return of the trajectory has been achieved
                    if score >= current_ret:
                        #print("demonstration return achieved: " + str(score))
                        break

    def save_counters(self):
        with open(self.args.hermes_name + "_updates.states", "w") as f:
            f.write(json.dumps(self.counter_update_states.tolist()))
        with open(self.args.hermes_name + "_starting.states", "w") as f:
            f.write(json.dumps(self.counter_starting_states.tolist()))
        with open(self.args.hermes_name + "_returns.cells", "w") as f:
            f.write(json.dumps(self.archive.returns.tolist()))
        with open(self.args.hermes_name + "_seen.cells", "w") as f:
            f.write(json.dumps(self.archive.seen.tolist()))
