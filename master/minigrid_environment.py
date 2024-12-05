"""
Implements our custom MiniGrid environment "DynObsDoor" which supports restoring states
Based on the "Dynamicobstaclesenv" environment
"""
import copy
from typing import Optional
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv
import numpy as np
from minigrid.core.world_object import Goal, Door, Ball
from gymnasium.spaces import Discrete
from operator import add
from minigrid.wrappers import FullyObsWrapper
import gymnasium as gym

# custom wrapper that returns a flattened image of the grid with the agent's position and number of time steps appended to it
class FlatFullyObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.env = env
        self.observation_space = env.observation_space.spaces["image"]

    def observation(self, observation):
        obs = observation["image"].flatten()
        obs = np.append(obs, self.env.steps)
        obs = np.append(obs, self.env.agent_dir)
        obs = np.append(obs, self.env.agent_pos[0])
        obs = np.append(obs, self.env.agent_pos[1])
        return obs

# base class for our custom MiniGrid environment
class MiniGridCustom(MiniGridEnv):
    # sets the starting position and direction of the agent
    # can either start in a specific position or randomly
    def set_start(self, x=None, y=None, d=None, random=False):
        if random:
            id_array = list(range(len(self.initial_states)))
            id = np.random.choice(id_array)
            self.agent_start_pos = self.initial_states[id]
            self.agent_start_dir = np.random.choice(list(range(0, 3)))
        else:
            self.agent_start_pos = (x, y)
            self.agent_start_dir = d

    # Translates numbers to action objects
    def num_to_action(self, action_num):
        action = None
        if action_num == 0:
            action = self.actions.left
        elif action_num == 1:
            action = self.actions.right
        elif action_num == 2:
            action = self.actions.forward
        elif action_num == 3:
            action = self.actions.pickup
        elif action_num == 4:
            action = self.actions.drop
        elif action_num == 5:
            action = self.actions.toggle
        elif action_num == 6:
            action = self.actions.done

        return action

    # returns a clone of the current environment
    def clone(self):
        return FlatFullyObsWrapper(FullyObsWrapper(copy.deepcopy(self)))

    def reset_to_state(self, state):
        pass

    def compress_state(self):
        pass

    # transform a compressed state into a list which is used for the maximal distance archive reduction strategy
    def compressed_to_list(self, compressed_state):
        list_state = []
        list_state.append(compressed_state[0])  # x
        list_state.append(compressed_state[1])  # y
        list_state.append(compressed_state[2])  # d

        if "obstacles" in compressed_state[3]:
            for obs_pos in compressed_state[3]["obstacles"]:
                list_state.append(obs_pos[0])  # x
                list_state.append(obs_pos[1])  # y
        if "doors" in compressed_state[3]:
            for door_pos in compressed_state[3]["doors"]:
                list_state.append(door_pos[2])  # is_open

        return list_state

# Dynamic Obstacles environment that support resetting to states
class MiniGridDynamicObstacles(MiniGridCustom):

    def __init__(
        self,
        n_obstacles,
        pos_r=1,
        neg_r=-1,
        step_r=0,
        #size=12,  # was 16
        #agent_start_pos=(1, 1),
        #agent_start_dir=0,
        max_steps: Optional[int] = None,
        **kwargs
    ):
        self.width = 12
        self.height = 8

        self.n_obstacles = int(n_obstacles)

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            # max_steps = 4 * self.width * self.height
            max_steps = 4 * (self.width-1) * (self.height-1)  # the outer part of the grid is not accessible anyway

        super().__init__(
            mission_space=mission_space,
            width=self.width,
            height=self.height,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs
        )
        # Allow only 3 actions permitted: left, right, forward
        self.action_space = Discrete(self.actions.forward + 1)
        self.reward_range = (neg_r, pos_r)

        # fully observable
        self.agent_view_size = max(self.width, self.height)
        self.highlight = False

        # initialize starting position
        self.agent_start_pos = (-1, -1)
        self.agent_start_dir = -1

        # generate grid and determine free cells
        self._gen_grid_no_obstacles()
        self.spawnable_positions = []
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                if self.grid.get(x, y) is None:
                    self.spawnable_positions.append((x, y))

        self.initial_states = [(1, 1), (2, 1), (3, 1)]
        # self.original_starters = self.spawnable_positions
        # self.original_starters = [(1, 1), (2, 1), (3, 1), (5, 1), (6, 1), (7, 1)]

        # initialize reward structure
        self.pos_r = pos_r
        self.neg_r = neg_r
        self.step_r = step_r

        # set size of observations (flattened image with agent coordinates appended to it)
        self.observation_size = 290 + 1 + 1  # + 1 BECAUSE WE ADDED THE NUMBER OF STEPS, was 387

        # number of steps since last reset
        self.steps = 0

        self.crashed = False

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    # generates a new grid without obstacles, just used to determine on which cells the agent can spawn
    def _gen_grid_no_obstacles(self):
        width = self.width
        height = self.height
        self.crashed = False

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        self.grid.vert_wall(4, 1, 4)

        self.grid.vert_wall(8, 1, 7)
        # Put holes in the wall
        self.grid.set(8, 1, None)
        self.grid.set(8, 2, None)

        #self.grid.vert_wall(12, 1, 7)
        # Put holes in the wall
        #self.grid.set(12, 5, None)
        #self.grid.set(12, 6, None)

        # Place a goal square in the bottem-right corner
        self.grid.set(width - 2, height - 2, Goal())

    # generates a new grid and places the agent at the stored starting position
    def _gen_grid(self, width, height):
        width = self.width
        height = self.height
        self.crashed = False

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        self.grid.vert_wall(4, 1, 4)
        self.grid.vert_wall(8, 1, 7)

        # Put holes in the wall
        self.grid.set(8, 1, None)
        self.grid.set(8, 2, None)

        #self.grid.vert_wall(12, 1, 7)
        # Put holes in the wall
        #self.grid.set(12, 5, None)
        #self.grid.set(12, 6, None)


        # Place a goal square in the bottom-right corner
        self.grid.set(width - 2, height - 2, Goal())

        # Place the agent
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

        # Place obstacles
        self.obstacles = []
        for i_obst in range(self.n_obstacles):
            self.obstacles.append(Ball())
            self.place_obj(self.obstacles[i_obst], max_tries=100)

        self.mission = "get to the green goal square"

        # initialize number of steps
        self.steps = 0

    def step(self, action):
        # assert that no action outside the action space was selected
        assert action < self.action_space.n


        # Check if there is an obstacle in front of the agent
        front_cell = self.grid.get(*self.front_pos)
        not_clear = front_cell and front_cell.type != "goal"

        # Update obstacle positions
        for i_obst in range(len(self.obstacles)):
            i = i_obst
            old_pos = self.obstacles[i_obst].cur_pos
            top = tuple(map(add, old_pos, (-1, -1)))

            try:
                self.place_obj(
                    self.obstacles[i_obst], top=top, size=(3, 3), max_tries=100
                )
                self.grid.set(old_pos[0], old_pos[1], None)
            except:
                pass



        # Update the agent's position/direction
        obs, reward, terminated, truncated, info = super().step(action)

        # Hack to overwrite the step reward of the original minigrid environment
        if reward == 0:
            reward = self.step_r

        # increase number of steps since last reset
        self.steps += 1

        # If the agent tried to walk over an obstacle or wall
        if action == self.actions.forward and not_clear:
            reward = self.neg_r
            terminated = True
            self.crashed = True
            return obs, reward, terminated, truncated, info

        return obs, reward, terminated, truncated, info


    # original MiniGrid implementation only uses this function for the positive reward, which is weird
    def _reward(self):
        return self.pos_r

    # a state is terminal if the agent cannot start on it -> walls and goals
    def terminal(self, x, y):
        if not (x, y) in self.spawnable_positions:
            return True
        else:
            return False

    # MOVE TO SELECTION STRATEGIES
    def get_obstacle_locations(self, grid_array):

        obstacle_locations = []

        for x in range(self.width):
            for y in range(self.height):
                if grid_array[x][y] == 6:
                    obstacle_locations.append((x, y))

        return obstacle_locations

    def get_grid_array(self, state):
        state = np.array(state[:-4])
        grid_array = state.reshape((self.width, self.height, 3))[:, :, 0]

        return grid_array

    def compress_state(self):
        x = self.agent_pos[0]
        y = self.agent_pos[1]
        d = self.agent_dir

        obstacle_positions = []
        for obs in self.obstacles:
            obstacle_positions.append(obs.cur_pos)

        objects = {}
        objects["obstacles"] = obstacle_positions

        return [x, y, d, objects]

    def reset_to_state(self, state):
        x, y, d, objects = state
        obstacle_positions = objects["obstacles"]

        # Reinitialize episode-specific variables
        self.agent_pos = (-1, -1)
        self.agent_dir = -1

        self._gen_grid_no_obstacles()

        self.agent_start_pos = (x, y)
        self.agent_pos = (x, y)
        self.agent_start_dir = d
        self.agent_dir = d

        self.obstacles = []
        for obs_pos in obstacle_positions:
            self.obstacles.append(Ball())
            self.grid.set(obs_pos[0], obs_pos[1], self.obstacles[-1])
            self.obstacles[-1].init_pos = obs_pos
            self.obstacles[-1].cur_pos = obs_pos

        try:
            # These fields should be defined
            assert (
                self.agent_pos >= (0, 0)
                if isinstance(self.agent_pos, tuple)
                else all(self.agent_pos >= 0) and self.agent_dir >= 0
            )
        except:
            print(self.agent_pos)
            print(self.agent_dir)
            print(self.__str__())
            raise ValueError("Invalid agent placement in reset to state!")

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        self.carrying = None

        self.crashed = False

        # Step count since episode start
        self.steps = 0
        self.step_count = 0

        if self.render_mode == "human":
            self.render()

        # Return first observation
        # obs = self.gen_obs()
        obs = self.grid.encode()

        obs = obs.flatten()   # THE WRAPPER DOES NOT WORK WHEN RESETTING TO STATE!
        obs = np.append(obs, self.steps)
        obs = np.append(obs, self.agent_dir)
        obs = np.append(obs, self.agent_pos[0])
        obs = np.append(obs, self.agent_pos[1])

        return obs, {}

# The DynObsDoor environment
class MiniGridDynamicObstaclesDoor(MiniGridCustom):

    def __init__(
        self,
        n_obstacles,
        pos_r=1,
        neg_r=-1,
        step_r=0,
        size=16,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: Optional[int] = None,
        **kwargs
    ):
        self.width = 16
        self.height = 8

        self.n_obstacles = int(n_obstacles)

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 200

        super().__init__(
            mission_space=mission_space,
            width=self.width,
            height=self.height,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs
        )
        # only 4 actions permitted: left, right, forward, toggle
        self.action_space = Discrete(4)
        self.reward_range = (neg_r, pos_r)

        # fully observable
        self.agent_view_size = max(self.width, self.height)
        self.highlight = False

        # initialize starting position
        self.agent_start_pos = (-1, -1)
        self.agent_start_dir = -1

        # generate grid and determine free cells
        self._gen_grid_no_obstacles()
        self.spawnable_positions = []
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                if self.grid.get(x, y) is None:
                    self.spawnable_positions.append((x, y))
        # agent can also spawn in the open door frame
        self.spawnable_positions.append((8, 2))

        self.initial_states = [(1, 1), (2, 1), (3, 1)]

        # initialize reward structure
        self.pos_r = pos_r
        self.neg_r = neg_r
        self.step_r = step_r

        # set size of observations (flattened image with agent coordinates appended to it)
        self.observation_size = 388

        # number of steps since last reset
        self.steps = 0

        self.crashed = False

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    # generates a new grid without obstacles, just used to determine on which cells the agent can spawn
    def _gen_grid_no_obstacles(self):
        width = self.width
        height = self.height
        self.crashed = False

        # create an empty grid
        self.grid = Grid(width, height)

        # generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        self.grid.vert_wall(4, 1, 4)
        self.grid.vert_wall(8, 1, 7)

        # place doors
        self.doors = [Door(color="yellow")]
        self.grid.set(8, 2, self.doors[-1])
        self.doors[-1].init_pos = (8, 2)
        self.doors[-1].cur_pos = (8, 2)

        self.grid.vert_wall(12, 1, 7)
        # put holes in the wall
        self.grid.set(12, 5, None)
        self.grid.set(12, 6, None)

        # place a goal square in the top-right corner
        self.grid.set(width - 2, 1, Goal())

    # generates a new grid and places the agent at the stored starting position
    def _gen_grid(self, width, height):
        self.crashed = False

        # Create an empty grid
        self.grid = Grid(self.width, self.height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, self.width, self.height)

        self.grid.vert_wall(4, 1, 4)
        self.grid.vert_wall(8, 1, 7)

        # Place doors
        self.doors = [Door(color="yellow")]
        self.grid.set(8, 2, self.doors[-1])
        self.doors[-1].init_pos = (8, 2)
        self.doors[-1].cur_pos = (8, 2)

        self.grid.vert_wall(12, 1, 7)
        # Put holes in the wall
        self.grid.set(12, 5, None)
        self.grid.set(12, 6, None)

        # Place a goal square in the top-right corner
        self.grid.set(self.width - 2, 1, Goal())

        # Place the agent
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

        # Place obstacles
        self.obstacles = []
        for i_obst in range(self.n_obstacles):
            self.obstacles.append(Ball())
            self.place_obj(self.obstacles[i_obst], max_tries=100)

        self.mission = "get to the green goal square"

        # initialize number of steps
        self.steps = 0

    def step(self, action):
        # assert that no action outside the action space was selected
        assert action < self.action_space.n
        if action == 3:  # toggle action
            action = 5

        # check if there is an obstacle in front of the agent
        front_cell = self.grid.get(*self.front_pos)

        not_clear = False
        # walking into a wall
        if front_cell and front_cell.type == "wall":
            not_clear = True
        # walking into a closed door
        elif front_cell and front_cell.type == "door":
            if not front_cell.is_open:
                not_clear = True
        # walking into an obstacle
        elif front_cell and front_cell.type == "ball":
            not_clear = True
        # nothing in front of agent (except for goal)
        else:
            not_clear = False

        # update obstacle positions
        for i_obst in range(len(self.obstacles)):
            old_pos = self.obstacles[i_obst].cur_pos
            top = tuple(map(add, old_pos, (-1, -1)))
            try:
                self.place_obj(
                    self.obstacles[i_obst], top=top, size=(3, 3), max_tries=100
                )
                self.grid.set(old_pos[0], old_pos[1], None)
            except:
                pass

        # update the agent's position/direction
        obs, reward, terminated, truncated, info = super().step(action)

        # hack to overwrite the step reward of the original minigrid environment
        if reward == 0:
            reward = self.step_r

        # increase number of steps since last reset
        self.steps += 1

        # if the agent tried to walk over an obstacle or wall
        if action == self.actions.forward and not_clear:
            reward = self.neg_r
            terminated = True
            self.crashed = True
            return obs, reward, terminated, truncated, info

        return obs, reward, terminated, truncated, info


    # original MiniGrid implementation only uses this function for the positive reward
    def _reward(self):
        return self.pos_r

    # a state is terminal if the agent cannot start on it -> walls and goals
    def terminal(self, x, y):
        if not (x, y) in self.spawnable_positions:
            return True
        else:
            return False

    def get_obstacle_locations(self, grid_array):
        obstacle_locations = []

        for x in range(self.width):
            for y in range(self.height):
                if grid_array[x][y] == 6:
                    obstacle_locations.append((x, y))

        return obstacle_locations

    # returns the environments current state in compressed form, which saves memory when building an archive
    def compress_state(self):
        x = self.agent_pos[0]
        y = self.agent_pos[1]
        d = self.agent_dir

        objects = {}

        obstacle_positions = []
        for obs in self.obstacles:
            obstacle_positions.append(obs.cur_pos)
        objects["obstacles"] = obstacle_positions

        door_positions = []
        for door in self.doors:
            door_x, door_y = door.cur_pos
            door_o = door.is_open
            door_positions.append((door_x, door_y, door_o))
        objects["doors"] = door_positions

        return [x, y, d, objects]

    # restore environment to a specific state given in compressed form
    def reset_to_state(self, state):
        x, y, d, objects = state
        obstacle_positions = objects["obstacles"]
        door_positions = objects["doors"]

        # reinitialize episode-specific variables
        self.agent_pos = (-1, -1)
        self.agent_dir = -1

        self._gen_grid_no_obstacles()

        self.agent_start_pos = (x, y)
        self.agent_pos = (x, y)
        self.agent_start_dir = d
        self.agent_dir = d

        self.obstacles = []
        for obs_pos in obstacle_positions:
            self.obstacles.append(Ball())
            self.grid.set(obs_pos[0], obs_pos[1], self.obstacles[-1])
            self.obstacles[-1].init_pos = obs_pos
            self.obstacles[-1].cur_pos = obs_pos

        self.doors = []
        for door_pos in door_positions:
            self.doors.append(Door(color="yellow", is_open=door_pos[2]))
            self.grid.set(door_pos[0], door_pos[1], self.doors[-1])
            self.doors[-1].init_pos = (door_pos[0], door_pos[1])
            self.doors[-1].cur_pos = (door_pos[0], door_pos[1])

        try:
            # these fields should be defined
            assert (
                self.agent_pos >= (0, 0)
                if isinstance(self.agent_pos, tuple)
                else all(self.agent_pos >= 0) and self.agent_dir >= 0
            )
        except:
            print(self.agent_pos)
            print(self.agent_dir)
            print(self.__str__())
            raise ValueError("Invalid agent placement in reset to state!")

        # check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        self.carrying = None

        self.crashed = False

        # step count since episode start
        self.steps = 0
        self.step_count = 0

        if self.render_mode == "human":
            self.render()

        # wrapper does not work when resetting to state
        obs = self.grid.encode()
        obs = obs.flatten()
        obs = np.append(obs, self.steps)
        obs = np.append(obs, self.agent_dir)
        obs = np.append(obs, self.agent_pos[0])
        obs = np.append(obs, self.agent_pos[1])

        return obs, {}
