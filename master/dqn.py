"""
Implementation of DQN
"""
import torch
from torch import optim
import torch.nn.functional as F
import random
from collections import deque
import matplotlib as mpl
import numpy as np
from rlmate.replay_buffer import ReplayBuffer as RB
import rlmate.util as util
import importlib
import logging
import racetrackgym.graphic as g
import json

logging.basicConfig(format="[%(asctime)s] %(message)s", level=logging.DEBUG)


class DQNAgent:
    def __init__(self, env, dqn_args):
        self.dqn_args = dqn_args
        self.seed = dqn_args.seed
        self.env = env
        self.benchmark = dqn_args.benchmark
        self.gamma = dqn_args.gamma

        # set seed
        random.seed(dqn_args.seed)
        np.random.seed(dqn_args.seed)
        torch.manual_seed(dqn_args.seed)

        # set device
        self.device = torch.device("cpu")
        device_string = "cpu"
        if self.dqn_args.gpu:
            if not torch.cuda.is_available():
                logging.warning("GPU option was set, but CUDA is not available")
            else:
                try:
                    device_string = "cuda:" + str(self.dqn_args.gpu_id)
                    self.device = torch.device(device_string)
                except:
                    logging.warning(
                        "Using "
                        + str(device_string)
                        + "did not work, does gpu_id exist?"
                    )
                    self.device_string = torch.device("cpu")
        logging.info("Device set to " + device_string)

        # initialize local and target Q-networks
        network_module = importlib.import_module(self.dqn_args.neural_network_file)
        if self.dqn_args.neural_network_weights == None:
            self.qnetwork_target = network_module.Network(self.device)
            self.qnetwork_local = network_module.Network(self.device)
        else:
            self.qnetwork_target = network_module.Network(
                self.device, self.dqn_args.neural_network_weights
            )
            self.qnetwork_local = network_module.Network(
                self.device, self.dqn_args.neural_network_weights
            )

        self.optimizer = optim.Adam(
            self.qnetwork_local.parameters(), lr=self.dqn_args.learning_rate
        )

        # initialize benchmark-specific variables
        if self.benchmark == "racetrack":
            self.env.random_start = False
            self.obs_size = 15
            self.action_space_dim = 9
            self.env_width = self.env.map.height
            self.env_height = self.env.map.width
            self.x_index = 0
            self.y_index = 1

        elif self.benchmark == "minigrid":
            self.obs_size = env.observation_size
            self.action_space_dim = self.env.action_space.n
            self.env_width = self.env.width
            self.env_height = self.env.height
            self.x_index = -2
            self.y_index = -1

        else:
            raise ValueError("Invalid benchmark!")

        # counters for visualizing starting states and states used for policy updates
        self.counter_starting_states = np.zeros((self.env_width, self.env_height))
        self.counter_update_states = np.zeros((self.env_width, self.env_height))

        # initialize replay buffer
        self.buffer = RB(
            self.dqn_args.buffer_size,
            1,
            [[self.obs_size]],
            self.dqn_args.batch_size,
            seed=self.seed,
            device=self.device,
        )

        # hyperparameters of learning:
        self.n_episodes = dqn_args.num_episodes
        self.eps_start = dqn_args.eps_start
        self.eps_end = dqn_args.eps_end
        self.eps_decay = dqn_args.eps_decay
        self.l_episodes = dqn_args.length_episodes

        # variables for learning
        self.best_score = dqn_args.best_network_score
        self.episodes_counter = 0
        self.update_counter = 0
        self.eps = self.eps_start
        self.means = []
        self.scores_window = deque(maxlen=100)

    # trains an agent according to DQN
    def train(self):
        # iterate for initialized number of episodes
        for _ in range(1, self.n_episodes + 1):
            self.episodes_counter += 1

            # reset state and score
            state = self.sample_initial_state()

            score = 0
            # make at most max_t steps
            for t in range(self.l_episodes):
                # act epsilon-greedy
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

            # update exploration epsilon
            self.eps = max(self.eps_end, self.eps_decay * self.eps)
            self.logging(score)

        self.save_network("end")
        self.save_counters()

        return self

    # sample from the set of initial states according to the benchmark
    def sample_initial_state(self):
        if self.benchmark == "racetrack":
            state = self.env.reset()
            x, y = state[:2]

        elif self.benchmark == "minigrid":
            self.env.set_start(random=True)
            state, _ = self.env.reset()
            x, y = self.env.agent_start_pos

        self.counter_starting_states[x][y] += 1
        return state

    # act epsilon greedy according to the local network
    def act(self, state, eps=0):
        state = torch.tensor(state).float().to(self.device)
        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        if random.random() > eps:
            res = np.argmax(action_values.cpu().numpy())
        else:
            res = random.choice(
                range(self.action_space_dim)
            )

        # Minigrid requires action objects and not numbers
        if self.benchmark == "minigrid":
            res = self.env.num_to_action(res)
        return res

    # take a step in the environment according to the benchmark
    def env_step(self, action):
        if self.benchmark == "racetrack":
            return self.env.step(action)
        elif self.benchmark == "minigrid":
            obs, reward, terminated, truncated, info = self.env.step(action)
            return reward, obs, terminated or truncated

    # add new sample to experience replay buffer
    def update_buffer(self, state, action, reward, next_state, done):
        # add the sample to the buffer
        state = np.array([state], dtype=object)
        next_state = np.array([next_state], dtype=object)

        self.buffer.add(state, action, reward, next_state, done)

    # update the local Q-network using samples from the experience replay buffer
    def learn(self, samples, gamma):
        states, actions, rewards, next_states, dones = samples
        states = states[0]
        next_states = next_states[0]

        # track location of states used for policy updates
        for state in states:
            x, y = int(state[self.x_index].item()), int(state[self.y_index].item())
            self.counter_update_states[x][y] += 1

        # implementation of DQN algorithm
        q_values_next_states = self.qnetwork_target.forward(next_states).max(dim=1)[0]
        targets = rewards + (gamma * (q_values_next_states) * (1 - dones))
        q_values = self.qnetwork_local.forward(states)

        actions = actions.view(actions.size()[0], 1)
        predictions = torch.gather(q_values, 1, actions).view(actions.size()[0])

        # calculate loss between targets and predictions
        loss = F.mse_loss(predictions, targets)

        # backward step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # perform a soft-update to the network
        for target_weight, local_weight in zip(
                self.qnetwork_target.parameters(), self.qnetwork_local.parameters()
        ):
            target_weight.data.copy_(
                self.dqn_args.tau * local_weight.data
                + (1.0 - self.dqn_args.tau) * target_weight.data
            )

    # periodically saves achieved returns and network weights
    def logging(self, score):
        self.scores_window.append(score)

        if self.episodes_counter % self.dqn_args.checkpoint_episodes == (
                -1 % self.dqn_args.checkpoint_episodes
        ):

            score = np.mean(self.scores_window)
            self.means.append(score)

            # if current score is better, save the network weights and update best seen score
            if score > self.best_score:
                self.best_score = score
                self.save_network("best")

            print(
                "\rEpisode {}\tAverage Score: {:.2f}\tBest Score: {:.2f}".format(
                    self.episodes_counter, score, self.best_score
                )
            )

            f = open(self.dqn_args.hermes_name + ".scores", "a")
            for score in self.scores_window:
                f.write(str(score) + "\n")
            f.close()

        if self.dqn_args.policy_extraction_frequency != 0 and (
                self.episodes_counter % self.dqn_args.policy_extraction_frequency
        ) == (-1 % self.dqn_args.policy_extraction_frequency):
            self.save_network(str(self.episodes_counter))

            if self.dqn_args.print_heatmaps:
                # creates heatmap showing which states the agent has visited and where it has started
                g.print_heatmap(
                    self.counter_update_states,
                    print_path=self.dqn_args.hermes_name + "_updates-heatmap_" + str(self.episodes_counter) + ".png",
                    show=False,
                    bounds=[0] + (np.array([0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])*1000000).tolist(),
                    colormap=mpl.cm.viridis
                )
                g.print_heatmap(
                    self.counter_starting_states,
                    print_path=self.dqn_args.hermes_name + "_started-heatmap_" + str(self.episodes_counter) + ".png",
                    show=False,
                    bounds=[0] + (np.array([0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) * 1000).tolist(),
                    colormap=mpl.cm.viridis
                )

    # saves the counters which can be used for visualization after training
    def save_counters(self):
        with open(self.dqn_args.hermes_name + "_updates.states", "w") as f:
            f.write(json.dumps(self.counter_update_states.tolist()))
        with open(self.dqn_args.hermes_name + "_starting.states", "w") as f:
            f.write(json.dumps(self.counter_starting_states.tolist()))

    def save_network(self, post_sign=None):
        path = self.dqn_args.hermes_name
        if post_sign != None:
            path += "_" + post_sign + ".pth"
        util.save_network(self.qnetwork_local, path=path)

    # used after init to load an existing network
    def load(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.qnetwork_local.load_state_dict(state_dict)
        self.qnetwork_target.load_state_dict(state_dict)

# creates a pseudo agent for evaluation
class Pseudo_agent:
    def __init__(self, network, name=None):
        self.network = network
        if name is None:
            self.name = "Pseudo Agent"
        else:
            self.name = name

    def get_name(self):
        return self.name

    def act(self, state=None):
        state = torch.tensor(state).float().to("cpu")
        with torch.no_grad():
            action_values = self.network(state)

        res = np.argmax(action_values.cpu().numpy())

        return res
