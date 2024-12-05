import torch
from rlmate.dict_replay_buffer import PrioritisedReplayBuffer
from rlmate.dict_replay_buffer import Variable as V
from dqn import DQNAgent

# prioritized replay buffer for DQNPR
class Buffer(PrioritisedReplayBuffer):
    def __init__(self, *args, beta: float = 1.0, min_prio: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self._beta = beta
        self._min_prio = min_prio

    def sample(self):
        return super().sample(self._beta)

    def add(self, state, action, reward, next_state, done) -> None:
        return super().add(
            state=state, action=action, reward=reward, next_state=next_state, done=done
        )

    def update_priorities(self, idxs, priorities):
        return super().update_priorities(idxs, [p + self._min_prio for p in priorities])


class DQNPRAgent(DQNAgent):
    def __init__(self, env, dqn_args):
        super().__init__(env, dqn_args)

        self.buffer = Buffer(
            self.dqn_args.buffer_size,
            self.dqn_args.batch_size,
            [
                V("state", shape=((self.obs_size,),)),
                V("action", dtype=torch.long),
                V("reward"),
                V("next_state", shape=((self.obs_size,),)),
                V("done"),
            ],
            seed=self.dqn_args.seed,
            alpha=self.dqn_args.pr_alpha,
            beta=self.dqn_args.pr_beta,
            min_prio=self.dqn_args.pr_min_prio,
            device=self.device,
        )

    # update the local Q-network using samples from the prioritized experience replay buffer and adjust sampling priorities
    def learn(self, samples, gamma):
        batch, is_weights, samples_ids = samples
        is_weights = is_weights.to(self.device)

        states, actions, rewards, next_states, dones = batch.unpack()

        for state in states:
            x, y = int(state[self.x_index].item()), int(state[self.y_index].item())
            self.counter_update_states[x][y] += 1

        dones = dones.squeeze(1)
        rewards = rewards.squeeze(1)

        q_values = self.qnetwork_local.forward(states)
        actions = actions.view(actions.size()[0], 1)
        predictions = torch.gather(q_values, 1, actions).view(actions.size()[0])

        with torch.no_grad():
            q_values_next_states = self.qnetwork_target.forward(next_states).max(dim=1)[
                0
            ]
            targets = rewards + (gamma * (q_values_next_states) * (1 - dones))

        td_error = targets - predictions
        weighted_td_residuals = is_weights * td_error.pow(2)
        loss = weighted_td_residuals.mean()
        new_priorities = torch.abs(td_error.detach())

        # make backward step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update priorities
        self.buffer.update_priorities(samples_ids, new_priorities.flatten().tolist())

        # perform a soft-update to the network
        for target_weight, local_weight in zip(
            self.qnetwork_target.parameters(), self.qnetwork_local.parameters()
        ):
            target_weight.data.copy_(
                self.dqn_args.tau * local_weight.data
                + (1.0 - self.dqn_args.tau) * target_weight.data
            )
