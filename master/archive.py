"""
Implementation of the archive used in RAREID and RAREPR
"""
import numpy as np
from scipy.spatial.distance import euclidean as euclidean


class Archive:
    def __init__(self, benchmark, env, initial_states_set, reduction_strategy, archive_size_factor, width, height):
        self.benchmark = benchmark
        self.env = env
        self.initial_states_set = initial_states_set
        self.width = width
        self.height = height
        self.reduction_strategy = reduction_strategy

        # the size of the archive is a fraction of the environment's free cells
        if self.benchmark == "racetrack":
            self.size = int((len(self.env.map.spawnable_positions) - len(self.initial_states_set)) * archive_size_factor)

        elif self.benchmark == "minigrid":
            self.size = int((len(self.env.spawnable_positions) - len(self.initial_states_set)) * archive_size_factor)

        else:
            raise ValueError("Invalid benchmark!")

        self.storage = [[[] for _ in range(self.height)] for _ in range(self.width)]

        self.reduced_storage = []

        self.ids_reduced_storage = []

    # add a state and its relevance to the archive
    def add_state(self, state, compressed_state, relevance):
        if self.benchmark == "racetrack":
            state = state[0:4]  # the remaining components of the state description are not relevant for the archive
            x, y, _, _ = state
            if not (x, y, 0, 0) in self.initial_states_set:
                self.storage[x][y].append([state, relevance])

        elif self.benchmark == "minigrid":
            x, y, d, _ = compressed_state  # compress state to save memory
            if (not (x, y, 0) in self.initial_states_set) and (not self.env.terminal(x, y)):
                self.storage[x][y].append([compressed_state, relevance])

    # returns a list containing all states in the reduced archive, relevances are only used internally
    def get_archived_states(self):
        return [x[0] for x in self.reduced_storage]

    # returns a single archived state at a given index
    def get_archived_state(self, index):
        return self.reduced_storage[index][0]

    # returns indices of all archived states
    def get_ids(self):
        return self.ids_reduced_storage

    # before reducing this archive we add all states from the previous archive (which is already reduced) into this archive
    # since they are likely still relevant
    def merge_archives(self, other_archive):
        # merging needs to be done before reducing this archive!
        assert self.reduced_storage == []

        # add all entries from the other reduced archive to this archive
        for archive_entry in other_archive.reduced_storage:
            x, y, _, _ = archive_entry[0]
            self.storage[x][y].append(archive_entry)

    # since we can only evaluate a fixed size of states, we reduce our archive to a fixed size by keeping only the most
    # relevant states that sufficiently cover the state space, this is done either using the cluster or the maximal
    # distance strategy
    def reduce_archive(self):
        if self.reduction_strategy == "cluster":
            # assign archived states in the same 2 by 2 grid neighborhood to the same clusters
            clusters = [[[] for _ in range(self.height)] for _ in range(self.width)]
            for x in range(self.width):
                for y in range(self.height):
                    if len(self.storage[x][y]) > 0:
                        for archive_entry in self.storage[x][y]:
                            clusters[x // 2][y // 2].append(archive_entry)
                        self.storage[x][y] = []

            # determine most relevant state in each cluster
            tmp = []
            for x in range(self.width):
                for y in range(self.height):
                    if len(clusters[x][y]) > 0:
                        tmp.append(max(clusters[x][y], key=lambda x: x[1]))
                        clusters[x][y] = []

            # reduce to fixed size and keep the most relevant states
            tmp = sorted(tmp, key=lambda x: x[1], reverse=True)
            # it could be that we have not gathered enough states to fill the archive
            sample_size = min(len(tmp), self.size)
            self.reduced_storage = tmp[0:sample_size]

        elif self.reduction_strategy == "maximal_distance":
            # for each map cell keep only the most relevant state
            tmp = []
            for x in range(self.width):
                for y in range(self.height):
                    if len(self.storage[x][y]) > 0:
                        most_relevant_state = max(self.storage[x][y], key=lambda x: x[1])
                        tmp.append(most_relevant_state)
                        self.storage[x][y] = []

            tmp = sorted(tmp, key=lambda x: x[1], reverse=True)
            # it could be that we have not gathered enough states to fill the archive
            sample_size = min(len(tmp), self.size)

            self.reduced_storage = [tmp[0]]
            for i in range(sample_size - 1):
                max_dist = 0
                # find the state with the largest minimum distance to all states that have already been added to
                # the reduced archive
                for archive_entry in tmp:
                    state = archive_entry[0]
                    if self.benchmark == "minigrid":
                        closest_distance = min(
                            euclidean(self.env.compressed_to_list(x[0]), self.env.compressed_to_list(state)) for x
                            in self.reduced_storage)
                    else:
                        closest_distance = min(euclidean(x[0], state) for x in self.reduced_storage)

                    if closest_distance > max_dist:
                        max_dist = closest_distance
                        max_dist_candidate = archive_entry

                self.reduced_storage.append(max_dist_candidate)
                tmp.remove(max_dist_candidate)

        self.ids_reduced_storage = np.arange(len(self.reduced_storage))
