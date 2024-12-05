'''
Implementation of RAREPR that is based on RAREID
'''
import torch
import numpy as np
import logging
from rareid import RAREIDAgent
import typing as t
from rlmate.dict_replay_buffer import Batch, PrioritisedReplayBuffer, Variable

logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.DEBUG)

# prioritized replay buffer for RAREPR
class RAREPRBuffer(PrioritisedReplayBuffer):
    def __init__(
        self,
        env,
        benchmark,
        size: int,
        batch_size: int,
        seed: t.Any = 0,
        device="cpu",
        progress_updates: int = 10,
    ):
        if benchmark == "racetrack":
            self._possible_starters = env.map.spawnable_positions
            state_space = Variable("state", shape=((15,),))
        elif benchmark == "minigrid":
            self._possible_starters = env.spawnable_positions
            state_space = Variable("state", shape=((env.observation_size,),))
        else:
            raise ValueError("Invalid benchmark!")

        variables = [
            state_space,
            Variable("action", dtype=torch.long),
            Variable("reward"),
            state_space.duplicate("next_state"),
            Variable("done"),
            Variable("starter", shape=((2,),), dtype=torch.long),
        ]
        super().__init__(size, batch_size, variables, seed, 1.0, device)
        self.progress_updates = progress_updates

        self._starter_priorities = {starter: 1.0 for starter in self._possible_starters}

        self._current_starter = ...
        self._current_starter_priority = ...

        self.benchmark = benchmark

    # saves the starting state of the current trajectory, used for computing priorities of all samples in the trajectory
    def set_current_starter(self, state):
        if self.benchmark == "racetrack":
            self._current_starter = (state[0], state[1])
            self._current_starter_priority = self._starter_priorities[self._current_starter]
        elif self.benchmark == "minigrid":
            self._current_starter = (state[-2], state[-1])
            self._current_starter_priority = self._starter_priorities[self._current_starter]

    # add a sample to the buffer
    def add(self, state, action, reward, next_state, done) -> None:
        self._max_priority = (
            self._current_starter_priority
        )
        return super().add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            starter=self._current_starter,
        )

    def _progress_message(self, idx: int):
        print("%.2f" % (100 * idx / self.num_stored), "%")

    # updates the buffer according to the evaluation regrets
    def update(self, evaluation_regrets_initial_states, evaluation_regrets_archived_states, psi):
        for k, v in evaluation_regrets_initial_states:
            self._starter_priorities[k] = v * (1.0 - psi)  # psi ensures that initial states have a sufficiently high priority

        for k, v in evaluation_regrets_archived_states:
            self._starter_priorities[k] = v * psi

    # samples a batch from the buffer
    def sample(self) -> t.Tuple[Batch, torch.Tensor, t.List]:
        batch, _, _ = super().sample(beta=1.0)
        state, action, reward, next_states, done, _ = batch.unpack()
        return (
            (state, None),
            action,
            reward.squeeze(1),
            (next_states, None),
            done.squeeze(1),
        )


class RAREPRAgent(RAREIDAgent):
    def __init__(self, env, dqn_args):
        super().__init__(env, dqn_args)

        # initialize RAREPR buffer
        self.buffer = RAREPRBuffer(
            env,
            self.dqn_args.benchmark,
            self.dqn_args.buffer_size,
            self.dqn_args.batch_size,
            self.dqn_args.seed,
            self.device,
        )

        # before the first evaluation stage, we only sample from the initial states
        self.sampling_probs = [0, 1.0]

        self.dictionary_mode = True

    # RAREPR samples an episode's starting state uniformly without any prioritization
    def sample_starting_state(self):
        # first sample with equal probability whether the starting state will be an initial state or an archived state
        if np.random.choice([0, 1], p=self.sampling_probs) == 1:
            initial_state_id = np.random.choice(self.ids_initial_states, p=self.starting_probs_initial_states)
            starter = self.initial_states[initial_state_id]
        else:
            archived_state_id = np.random.choice(self.current_archive.get_ids())  # sample uniformly from the archive
            starter = self.current_archive.get_archived_state(archived_state_id)

        self.counter_starting_states[starter[0]][starter[1]] += 1

        return starter

    # update RAREPR's training priorities by setting the replay buffer's priorities according to the evaluation regrets
    def update_priorities(self, evaluation_regrets_initial_states, evaluation_regrets_archived_states, archived_states, psi):
        self.buffer.update(evaluation_regrets_initial_states, evaluation_regrets_archived_states, psi)

        # after first evaluation stage, we either sample from the initial states or the archived states with equal probability
        if self.sampling_probs == [0, 1.0]:
            self.sampling_probs = [0.5, 0.5]

    def restore_state(self, state):
        restored_state = super().restore_state(state)
        self.buffer.set_current_starter(restored_state)

        return restored_state
