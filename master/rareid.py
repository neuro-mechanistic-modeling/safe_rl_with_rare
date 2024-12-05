'''
Implementation of RAREID that is based on DQN
'''
import matplotlib as mpl
import torch
from torch import nn
import numpy as np
import logging
import json
from dqn import DQNAgent
from evaluation_stage import EvaluationStage
import racetrackgym.graphic as g
from archive import Archive

logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.DEBUG)

class RAREIDAgent(DQNAgent):
    def __init__(self, env, dqn_args):
        super().__init__(env, dqn_args)
        self.args = dqn_args

        # store environment's initial states
        self.initial_states = []
        self.initial_states_set = set()
        if self.benchmark == "racetrack":
            for starter in self.env.map.starters:
                starter = (starter[0], starter[1], 0, 0)
                self.initial_states.append(starter)
                self.initial_states_set.add(starter)

        elif self.benchmark == "minigrid":
            for starter in self.env.initial_states:
                x, y = starter
                self.initial_states.append((x, y, 0))
                self.initial_states_set.add((x, y, 0))

        else:
            raise ValueError("Invalid benchmark!")

        # initialize storage for the best evaluation values which is used in regret estimation
        self.best_evals = np.ones((self.env_width, self.env_height)) * -1

        # initialize the probabilities for sampling a starting state, initially it is a uniform distribution
        # over the initial states
        self.starting_probs_initial_states = np.ones(len(self.initial_states)) / len(self.initial_states)
        self.starting_probs_archived_states = []

        # store indices of initial states for sampling
        self.ids_initial_states = np.arange(len(self.initial_states))

        # psi is initially 0 since no archived states are available
        self.psi = 0.0

        # store the best evaluation in the initial states
        self.best_eval = 0.0

        # initialize the archives
        self.current_archive = Archive(benchmark=self.args.benchmark, env=self.env, initial_states_set=self.initial_states_set,
                                       reduction_strategy=self.args.reduction_strategy, archive_size_factor=self.args.archive_size_factor,
                                       width=self.env_width, height=self.env_height)
        self.next_archive = Archive(benchmark=self.args.benchmark, env=self.env, initial_states_set=self.initial_states_set,
                                       reduction_strategy=self.args.reduction_strategy, archive_size_factor=self.args.archive_size_factor,
                                       width=self.env_width, height=self.env_height)

        # counters for visualizing archived states
        self.counter_archived_states = np.zeros((self.env_width, self.env_height))

        self.relevance_heuristic = self.args.relevance_heuristic
        # if we use the novelty relevance heuristic, we need to initialize the predictor and target networks of RND
        if self.relevance_heuristic == "novelty":
            self.init_rnd()

        self.dictionary_mode = False

    # trains RAREID agent by alternating between learning and evaluation stages
    def train(self):
        for _ in range(1, self.n_episodes + 1):
            self.episodes_counter += 1

            self.learning_stage()

            if self.episodes_counter >= self.args.es_pre_training and (
                    self.episodes_counter - self.args.es_pre_training) % self.args.es_length_stage == 0:
                self.evaluation_stage()

        self.save_network('end')
        self.save_counters()
        return self

    # executes multiple learning stages in which the agent is trained and relevant states are archived
    def learning_stage(self):
        ### STEP 1: SAMPLE INITIAL OR ARCHIVED STATE ###
        starter = self.sample_starting_state()

        ### STEP 2: RESTORE STATE ###
        state = self.restore_state(starter)

        ### STEP 3: GENERATE EXPERIENCES ###
        score = 0
        compressed_state = None
        for t in range(self.l_episodes):
            action = self.act(state, self.eps)

            # in MiniGrid we need to store a compressed version of the state to save memory
            if self.benchmark == "minigrid":
                compressed_state = self.env.compress_state()

            reward, next_state, done = self.env_step(action)  # send the action to the environment and observe

            # add the sample to the buffer
            self.update_buffer(state, action, reward, next_state, done)

            ### STEP 4: DETERMINE RELEVANCE OF STATES AND EXPAND ARCHIVE ###
            relevance = self.determine_relevance(state, next_state, reward)
            self.next_archive.add_state(state, compressed_state, relevance)

            ### STEP 5: LEARN FROM EXPERIENCES ###
            self.update_counter = (self.update_counter + 1) % self.dqn_args.update_every
            if self.update_counter == 0:
                if (len(self.buffer)) > self.dqn_args.batch_size:
                    samples = self.buffer.sample()
                    self.learn(samples, self.gamma)

                    # update RND predictor network if we use the novelty relevance heuristic
                    if self.relevance_heuristic == "novelty":
                        self.update_rnd(samples)

            state = next_state
            score += reward * np.power(self.gamma, t)
            if done:
                break

        # update exploration epsilon
        self.eps = max(self.eps_end, self.eps_decay * self.eps)
        self.logging(score)

    # executes an evaluation stage in which the agent is evaluated on all initial and archived states and the training
    # priorities are adapted accordingly
    def evaluation_stage(self):
        logging.info(f"Running Evaluation stage {self.episodes_counter}")
        # merge next with current archive since old archived states might still be relevant
        self.next_archive.merge_archives(self.current_archive)

        ### STEP 1: REDUCE THE NEXT ARCHIVE ###
        self.next_archive.reduce_archive()
        archived_states = self.next_archive.get_archived_states()

        # update counter for archived states:
        for state in archived_states:
            self.counter_archived_states[state[0]][state[1]] += 1

        ### STEP 2: EVALUATE ALL INITIAL AND ARCHIVED STATES ###
        evaluation_stage_initial_states = EvaluationStage(
            self, self.args, self.env, self.episodes_counter, starters=self.initial_states, grps_mode=self.args.es_grp
        )
        evaluation_stage_initial_states.eval()
        # update psi with evaluation values of initial states
        self.psi = evaluation_stage_initial_states.get_psi()

        evaluation_stage_archived_states = EvaluationStage(
            self, self.args, self.env, self.episodes_counter, starters=archived_states, grps_mode=self.args.es_grp
        )
        evaluation_stage_archived_states.eval()

        if self.args.print_heatmaps:
            # prints heatmap showing regrets of the recently evaluated archived states
            g.print_heatmap(evaluation_stage_archived_states.get_regrets(self.best_evals),
                            print_path=self.args.hermes_name + "_regret-heatmap_" + str(self.episodes_counter) + ".png",
                            show=False)

        ### STEP 3: APPROXIMATE REGRET ###
        if not self.args.no_regret:
            evaluation_regrets_initial_states, self.best_evals = evaluation_stage_initial_states.evaluation_regrets(
                self.best_evals, self.dictionary_mode)

            evaluation_regrets_archived_states, self.best_evals = evaluation_stage_archived_states.evaluation_regrets(
                self.best_evals, self.dictionary_mode)

        else:  # turn off regret approximation for ablation study
            print("\n")
            print("using no regret approximation!")
            evaluation_regrets_initial_states = evaluation_stage_initial_states.priorities(self.dictionary_mode)

            evaluation_regrets_archived_states = evaluation_stage_archived_states.priorities(self.dictionary_mode)

        ### STEP 4: UPDATE TRAINING PRIORITIES ###
        self.update_priorities(evaluation_regrets_initial_states, evaluation_regrets_archived_states, archived_states, self.psi)

        # initialize new archive
        self.current_archive = self.next_archive
        self.next_archive = Archive(benchmark=self.args.benchmark, env=self.env, initial_states_set=self.initial_states_set,
                                       reduction_strategy=self.args.reduction_strategy, archive_size_factor=self.args.archive_size_factor,
                                       width=self.env_width, height=self.env_height)

        if self.args.print_heatmaps:
            # prints heatmap showing the locations of all archived states
            g.print_heatmap(
                self.counter_archived_states,
                print_path=self.args.hermes_name + "_archived-heatmap_" + str(self.episodes_counter) + ".png",
                show=False,
                bounds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                colormap=mpl.cm.viridis
            )
            # prints heatmap showing the evaluation values of the recently evaluated archived states
            if self.args.es_grp:
                g.print_heatmap(evaluation_stage_archived_states.get_grps(),
                                print_path=self.args.hermes_name + "_grp-heatmap_" + str(self.episodes_counter) + ".png",
                                show=False)
            else:
                g.print_heatmap(evaluation_stage_archived_states.get_returns(),
                                print_path=self.args.hermes_name + "_return-heatmap_" + str(self.episodes_counter) + ".png",
                                bounds=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 0.9, 1],
                                show=False)

    # RAREID samples an episode's starting state according to the training priorities
    def sample_starting_state(self):
        # first sample whether the starting state will be an initial state or an archived state, and then sample the
        # starting state according to a distribution over the regret
        if np.random.choice([0, 1], p=[self.psi, 1-self.psi]) == 1:
            initial_state_id = np.random.choice(self.ids_initial_states, p=self.starting_probs_initial_states)
            starter = self.initial_states[initial_state_id]
        else:
            archived_state_id = np.random.choice(self.current_archive.get_ids(), p=self.starting_probs_archived_states)
            starter = self.current_archive.get_archived_state(archived_state_id)

        self.counter_starting_states[starter[0]][starter[1]] += 1  # TODO: DOES THIS WORK FOR BOTH BENCHMARKS?

        return starter

    # restore the environment's state to any given state
    def restore_state(self, state):
        # restore the state according to the benchmark
        if self.benchmark == "racetrack":
            x, y, a, b = state
            self.env.reset_to_state((x, y), (a, b))
            return self.env.get_state()

        elif self.benchmark == "minigrid":
            # if we restart in an initial state we only care about position and direction
            if len(state) == 3:
                x, y, d = state
                # set starting state
                self.env.set_start(x=x, y=y, d=d)
                # reset grid
                return self.env.reset()[0]
            # when restarting in an archived state we want to restore the exact state
            else:
                return self.env.reset_to_state(state)[0]
        else:
            raise ValueError("Invalid benchmark!")

    # determines the relevance of a given state according to the value or novelty heuristic
    def determine_relevance(self, state, next_state, reward):
        if self.relevance_heuristic == "value":
            curr_value = self.state_value(state)
            next_value = self.state_value(next_state)

            return np.abs(curr_value - (next_value + reward))
            # return curr_value - next_value

        elif self.relevance_heuristic == "novelty":
            curr_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            rnd_prediction = self.predictor_rnd(curr_tensor).detach()
            rnd_target = self.target_rnd(curr_tensor).detach()

            return (rnd_prediction - rnd_target).pow(2).mean().numpy()

    # update RAREID's training priorities by defining starting states' probability distributions over the evaluation regrets
    def update_priorities(self, evaluation_regrets_initial_states, evaluation_regrets_archived_states, archived_states, psi):
        # convert priorities to probabilities
        probabilities_initial_states_2d = evaluation_regrets_initial_states / evaluation_regrets_initial_states.sum()
        # convert 2D array to 1D array
        probabilities_initial_states_1d = np.zeros(len(self.initial_states))
        for i, state in enumerate(self.initial_states):
            probabilities_initial_states_1d[i] = probabilities_initial_states_2d[state[0], state[1]]
        # update starting probabilities of initial states
        self.starting_probs_initial_states = np.array(probabilities_initial_states_1d)

        # do the same for archived states
        probabilities_archived_states_2d = evaluation_regrets_archived_states / evaluation_regrets_archived_states.sum()
        probabilities_archived_states_1d = np.zeros(len(archived_states))
        for i, state in enumerate(archived_states):
            probabilities_archived_states_1d[i] = probabilities_archived_states_2d[state[0], state[1]]
        self.starting_probs_archived_states = np.array(probabilities_archived_states_1d)

    def save_counters(self):
        super().save_counters()
        with open(self.args.hermes_name + "_archived.states", "w") as f:
            f.write(json.dumps(self.counter_archived_states.tolist()))


    # initializes the target and predictor networks of random network distillation (RND), the predictor network's loss
    # will be used to approximate novelty
    def init_rnd(self):
        assert (self.relevance_heuristic == "novelty")
        torch.manual_seed(self.seed)

        if self.benchmark == "racetrack":
            i = 15
            x = 64
        elif self.benchmark == "minigrid":
            i = self.env.observation_size
            x = 128

        self.target_rnd = nn.Sequential(
            nn.Linear(i, x),
            nn.ReLU(),
            nn.Linear(x, x),
            nn.ReLU(),
            nn.Linear(x, x)
        )
        self.predictor_rnd = nn.Sequential(
            nn.Linear(i, x),
            nn.ReLU(),
            nn.Linear(x, x),
            nn.ReLU(),
            nn.Linear(x, x)
        )
        self.rnd_optim = torch.optim.SGD(self.predictor_rnd.parameters(), lr=8e-4)

    # update RND's predictor network using the same samples that are used to update the Q-network
    def update_rnd(self, samples):
        states = samples[0]
        states = states[0]

        rnd_targets = self.target_rnd(states).flatten()
        rnd_predictions = self.predictor_rnd(states).flatten()

        rnd_loss = (rnd_predictions - rnd_targets.detach()).pow(2).mean()

        self.rnd_optim.zero_grad()
        rnd_loss.backward()
        self.rnd_optim.step()

    # recover state value from Q-values
    def state_value(self, state):
        state = torch.tensor(state).float().to(self.device)
        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        action_values = action_values.cpu().numpy()

        greedy_action = np.argmax(action_values)

        fraction = self.eps / self.action_space_dim

        value = action_values[greedy_action] * (1.0 - self.eps + fraction)

        for i in range(self.action_space_dim):
            if i == greedy_action:
                continue
            else:
                value += action_values[i] * fraction

        return value
