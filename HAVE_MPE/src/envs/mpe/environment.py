import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from .multi_discrete import MultiDiscrete
import copy
import time

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(
        self,
        world,
        reset_callback=None,
        reward_callback=None,
        observation_callback=None,
        state_callback=None,
        info_callback=None,
        world_info_callback=None,
        done_callback=None,
        shared_viewer=True,
        time_limit=25,
    ):

        world = copy.deepcopy(world)
        self.world = world
        self.agents = self.world.policy_agents
        self.preys = self.world.policy_preys
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.state_callback = state_callback
        self.info_callback = info_callback
        self.world_info_callback = world_info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.range_p = world.range_p
        self.dim_p = world.dim_p

        self.time = 0
        self.time_limit = time_limit

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

        self.action_space = spaces.Tuple(tuple(self.action_space))
        self.observation_space = spaces.Tuple(tuple(self.observation_space))
        self.n_agents = self.n

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def seed(self, seed):
        self.world.seed(seed)

    def bound(self, x):
        d = np.zeros(2)
        if abs(x[0])>abs(x[1]) and x[0]<0 and abs(x[0])>0.8*self.range_p:
            d[0] = 2
        if abs(x[0])>abs(x[1]) and x[0]>0 and abs(x[0])>0.8*self.range_p:
            d[0] = -2
        if abs(x[0])<abs(x[1]) and x[1]<0 and abs(x[1])>0.8*self.range_p:
            d[1] = 2
        if abs(x[0])<abs(x[1]) and x[1]>0 and abs(x[1])>0.8*self.range_p:
            d[1] = -2
        return d

    def step(self, action_n):
        one_hot_actions = []
        for act, acsp in zip(action_n, self.action_space):
            one_hot = np.zeros(acsp.n)
            one_hot[act] = 1.0
            one_hot_actions.append(one_hot)

        action_n = one_hot_actions

        obs_n = []
        reward_n = []
        done_n = []
        info = {'n': []}
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # set action for each prey
        for j, prey in enumerate(self.preys):
            self._set_prey_action(None, prey, self.action_space[0])

        # advance world state
        self.world.step()
        # record observation for each agent
        for i, agent in enumerate(self.agents):
            obs_n.append(self._get_obs(agent))
            # if not self.shared_reward:
            #     reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))

        # if self.shared_reward:
        #     reward_n.append(self._get_reward(agent))
        reward_n.append(self._get_reward(agent))
        info['n'].append(self._get_info(agent))

        self.time += 1
        return tuple(obs_n), reward_n, done_n, info

    def reset(self):
        # restart time step count
        self.time = 0
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))

        return tuple(obs_n)

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    def get_state(self):
        # same as get_obs_full but public
        if self.state_callback is None:
            return None
        return self.state_callback(self.world)

    # get world info for benchmarking
    def _get_world_info(self):
        if self.world_info_callback is None:
            return {}
        return self.world_info_callback(self.world, final=self.time==self.time_limit)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world).astype(np.float32)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
        else:
            action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                else:
                    agent.action.u = action[0]
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    def _set_prey_action(self, action, prey, action_space, time=None):
        prey_action = np.zeros(action_space.n)
        min_dist = 10000
        direction = []
        # construct action: move following the oppisite direction of closest agent
        for agent in self.agents:
            dist = np.sqrt(np.sum(np.square(prey.state.p_pos - agent.state.p_pos)))
            if dist < min_dist:
                min_dist = dist
                direction = (prey.state.p_pos - agent.state.p_pos) / dist
                direction_intensity = np.abs(direction)
                direction[np.argmax(direction_intensity)] = np.sign(direction[np.argmax(direction_intensity)]) * 1
                direction[np.argmin(direction_intensity)] = 0
        # not allow to cross the boundary
        in_bound = self.bound(prey.state.p_pos)
        prey_action[1] = direction[0] + in_bound[0]
        prey_action[3] = direction[1] + in_bound[1]
        # if captured, prey chooses to stay
        if min_dist <= (prey.size + agent.size):
            prey_action[0] = 1
            prey_action[1] = 0
            prey_action[3] = 0
        self.force_discrete_action = False
        self._set_action(prey_action, prey, self.action_space[0])
        self.force_discrete_action = True

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            # print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from envs.mpe import rendering
                self.viewers[i] = rendering.Viewer(700,700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from envs.mpe import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from envs.mpe import rendering
            # update bounds to center around agent
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))
            # save picture
            # import matplotlib.pyplot as plt
            # x = self.viewers[i].render(return_rgb_array = mode=='rgb_array')
            # plt.imshow(x)
            # plt.axis('off')
            # plt.gca().set_axis_off()
            # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            # plt.margins(0, 0)
            # plt.savefig('cooperative_navigation_5.svg', format='svg', bbox_inches='tight', pad_inches=0)
            # plt.savefig('cooperative_navigation_5.pdf', bbox_inches='tight', pad_inches=0)
            # plt.savefig('cooperative_navigation_5.jpg', bbox_inches='tight', pad_inches=0)
            # plt.savefig('cooperative_navigation_20.svg', format='svg', bbox_inches='tight', pad_inches=0)
            # plt.savefig('cooperative_navigation_20.pdf', bbox_inches='tight', pad_inches=0)
            # plt.savefig('cooperative_navigation_20.jpg', bbox_inches='tight', pad_inches=0)
            # plt.savefig('predator_prey_5.svg', format='svg', bbox_inches='tight', pad_inches=0)
            # plt.savefig('predator_prey_5.pdf', bbox_inches='tight', pad_inches=0)
            # plt.savefig('predator_prey_5.jpg', bbox_inches='tight', pad_inches=0)
            # plt.savefig('predator_prey_20.svg', format='svg', bbox_inches='tight', pad_inches=0)
            # plt.savefig('predator_prey_20.pdf', bbox_inches='tight', pad_inches=0)
            # plt.savefig('predator_prey_20.jpg', bbox_inches='tight', pad_inches=0)
            # print('save picture!!')

        if self.shared_viewer:
            assert len(results) == 1
            return results[0]

        return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x,y]))
        return dx

    def close(self):
        for viewer in self.viewers:
            if viewer:
                viewer.close()


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n
