import numpy as np
from ..core import World, Agent, Landmark
from ..scenario import BaseScenario
import random

# Fully Observation: Observe all preys and agents in environments
# Shared reward
class Scenario(BaseScenario):
    def make_world(
            self,
            num_agents=5,
            num_landmarks=5,
            num_agents_obs=4,
            num_landmarks_obs=5
    ):
        world = World()
        # set any world properties first
        world.dim_c = 0
        num_preys = num_landmarks
        num_preys_obs = num_landmarks_obs
        world.num_agents = num_agents
        world.num_preys = num_preys
        world.num_agents_obs = num_agents_obs
        world.num_preys_obs = num_preys_obs

        world.collaborative = True
        world.discrete_action = True

        world.agents_infos = np.zeros((num_agents, 6)) # position(2) + velocity(2) + size(1) + observe capacity (1, MAX=2*1.414=2.828)
        world.preys_infos = np.zeros((num_preys, 5)) # position(2) + velocity(2) + size(1)

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            world.agents_infos[i, 4] = 0.04 # size
            agent.size = world.agents_infos[i, 4]
            world.agents_infos[i, 5] = 2.0 # observe distance
            agent.accel = 5.0

        # add preys
        world.preys = [Agent() for i in range(num_preys)]
        for i, prey in enumerate(world.preys):
            prey.name = 'prey %d' % i
            prey.collide = False
            prey.movable = True
            prey.silent = True
            world.preys_infos[i, 4] = 0.08 # size
            prey.size = world.preys_infos[i, 4]
            prey.accel = 7.0

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for preys
        for i, prey in enumerate(world.preys):
            prey.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for i, agent in enumerate(world.agents):
            world.agents_infos[i, :2] = world.np_random.uniform(-world.range_p, +world.range_p, world.dim_p)
            world.agents_infos[i, 2:4] = np.zeros(world.dim_p)
            agent.state.p_pos = world.agents_infos[i, :2]
            agent.state.p_vel = world.agents_infos[i, 2:4]
            agent.state.c = np.zeros(world.dim_c)

        for i, prey in enumerate(world.preys):
            world.preys_infos[i, :2] = world.np_random.uniform(-world.range_p, +world.range_p, world.dim_p)
            world.preys_infos[i, 2:4] = np.zeros(world.dim_p)
            prey.state.p_pos = world.preys_infos[i, :2]
            prey.state.p_vel = world.preys_infos[i, 2:4]

    def benchmark_data(self, agent, world):
        rew_n = 0
        collisions_n = 0
        catched_preys_n = 0

        agents_pos = world.agents_infos[:, :2] # position(2) + velocity(2) + size(1) + observe capacity(1)
        preys_pos = world.preys_infos[:, :2] # position(2) + velocity(2) + size(1)
        agents_size = world.agents_infos[:, 4] # (n_agents,)
        preys_size = world.preys_infos[:, 4] # (n_preys,)

        expand_pos_1 = np.expand_dims(preys_pos, axis=1) # (n_preys, 2) => (n_preys, 1, 2)
        expand_pos_2 = np.expand_dims(agents_pos, axis=0) # (n_agents, 2) => (1, n_agents, 2)
        expand_pos_3 = np.expand_dims(agents_pos, axis=1) # (n_agents, 2) => (n_agents, 1, 2)

        # compute relative distance
        prey_diff_agent = expand_pos_1 - expand_pos_2 # (n_preys, n_agents, 2) (prey - other agents)
        agent_diff_agent = expand_pos_2 - expand_pos_3 # (n_agents, n_agents, 2) (other agents - agent)
        prey_dist_agent = np.linalg.norm(prey_diff_agent, axis=-1)  # (n_preys, n_agents)
        agent_dist_agent = np.linalg.norm(agent_diff_agent, axis=-1)  # (n_agents, n_agents)

        # occupied distance
        catched_dist = np.expand_dims(agents_size, axis=0) + np.expand_dims(preys_size, axis=1) # (n_preys, n_agents)
        # collision distance
        collision_dist = np.expand_dims(agents_size, axis=1) + np.expand_dims(agents_size,
                                                                             axis=0)  # (n_agents, n_agents)

        collision_mask = agent_dist_agent<collision_dist

        collisions_n += (collision_mask.sum()-world.num_agents) / 2
        min_dists = np.min(prey_dist_agent, axis=-1) # (n_preys,)
        catched_preys_n += (np.sum(prey_dist_agent < catched_dist, axis=-1) > 0).sum()

        min_dists_n = min_dists.sum()
        rew_n -= min_dists_n
        rew_n -= collisions_n

        info = {}
        info['rew'] = rew_n
        info['collisions'] = collisions_n
        info['min_dists'] = min_dists_n
        info['catch_preys'] = catched_preys_n
        return info

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        collision_dist = agent1.size + agent2.size
        return True if dist < collision_dist else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each prey, penalized for collisions
        rew_n = 0
        # global reward
        collisions_n = 0

        agents_pos = world.agents_infos[:, :2] # position(2) + velocity(2) + size(1) + observe capacity(1)
        preys_pos = world.preys_infos[:, :2] # position(2) + velocity(2) + size(1)
        agents_size = world.agents_infos[:, 4] # (n_agents,)

        expand_pos_1 = np.expand_dims(preys_pos, axis=1) # (n_preys, 2) => (n_preys, 1, 2)
        expand_pos_2 = np.expand_dims(agents_pos, axis=0) # (n_agents, 2) => (1, n_agents, 2)
        expand_pos_3 = np.expand_dims(agents_pos, axis=1) # (n_agents, 2) => (n_agents, 1, 2)

        # compute relative distance
        prey_diff_agent = expand_pos_1 - expand_pos_2  # (n_preys, n_agents, 2)
        agent_diff_agent = expand_pos_2 - expand_pos_3 # (n_agents, n_agents, 2)
        prey_dist_agent = np.linalg.norm(prey_diff_agent, axis=-1)  # (n_preys, n_agents)
        agent_dist_agent = np.linalg.norm(agent_diff_agent, axis=-1)  # (n_agents, n_agents)

        collision_dist = np.expand_dims(agents_size, axis=1) + np.expand_dims(agents_size,
                                                                              axis=0)  # (n_agents, n_agents)

        collision_mask = agent_dist_agent < collision_dist

        collisions_n += (collision_mask.sum() - world.num_agents) / 2
        min_dists = np.min(prey_dist_agent, axis=-1)
        min_dists_n = min_dists.sum()

        rew_n -= min_dists_n + collisions_n
        return rew_n

    def observation(self, agent, world):
        for i, other in enumerate(world.agents):
            if i==0 and other is agent:
                self.generate_observation(world)
            if agent is other:
                return world.obs_n[i]

    def generate_observation(self, world):
        agents_pos = world.agents_infos[:, :2] # position(2) + velocity(2) + size(1)
        agents_vel = world.agents_infos[:, 2:4] # position(2) + velocity(2) + size(1)
        preys_pos = world.preys_infos[:, :2] # position(2) + velocity(2) + size(1)
        expand_pos_1 = np.expand_dims(agents_pos, axis=1) # (n_agents, 2) => (n_agents, 1, 2)
        expand_pos_2 = np.expand_dims(agents_pos, axis=0) # (n_agents, 2) => (1, n_agents, 2)
        expand_pos_3 = np.expand_dims(preys_pos, axis=0) # (n_preys, 2) => (1, n_preys, 2)

        # compute relative position
        agent_diff_agent = expand_pos_2 - expand_pos_1 # (n_agents, n_agents, 2) (other agents - agent)
        agent_diff_prey = expand_pos_3 - expand_pos_1 # (n_preys, n_preys, 2) (preys - agent)

        # 1. observe other agents and preys in specific nearest number
        # 1.1. compute relative distance
        # agent_dist_agent = np.linalg.norm(agent_diff_agent, axis=-1, keepdims=True)  # (n_agents, n_agents, 1)
        # agent_dist_prey = np.linalg.norm(agent_diff_prey, axis=-1, keepdims=True)  # (n_agents, n_preys, 1)
        # 1.2. sort by relative distance
        # sorted_agent_indices = np.argsort(agent_dist_agent, axis=1)[:, 1:] # (n_agents, n_agents-1, 1), exclude self
        # sorted_prey_indices = np.argsort(agent_dist_prey, axis=1) # (n_agents, n_preys, 1)
        # 1.3. select nearest the agents and preys
        # obs_agents = np.take_along_axis(agent_diff_agent, sorted_agent_indices, axis=1)[:, :world.num_agents_obs]
        # obs_preys = np.take_along_axis(agent_diff_prey, sorted_prey_indices, axis=1)[:, :world.num_preys_obs]

        # 2.observe other agents and preys in multi-agent environments
        obs_agents = agent_diff_agent # (n_agents, n_agents, 2)
        obs_preys = agent_diff_prey # (n_agents, n_preys, 2)
        obs_agents = obs_agents[np.eye(world.num_agents) == 0].reshape(world.num_agents, -1, 2) # (n_agents, n_agents-1, 2)

        # flatten
        obs_agents_flatten = obs_agents.reshape(world.num_agents, -1)
        obs_preys_flatten = obs_preys.reshape(world.num_agents, -1)

        obs_n = np.concatenate((agents_vel, agents_pos, obs_agents_flatten, obs_preys_flatten), axis=-1)
        world.obs_n = obs_n
        return obs_n

    ################## For TransfQMIX ##################
    def entity_observation(self, agent, world):
        for i, other in enumerate(world.agents):
            if i==0 and other is agent:
                self.generate_entity_observation(world)
            if agent is other:
                return world.obs_n[i]

    # entity_obs
    def generate_entity_observation(self, world):

        """
        Entity approach for transformers

        *4 features for each entity: (rel_x, rel_y, is_agent, is_self)*
        - rel_x and rel_y are the relative positions of an entity in respect to agent
        - communication is not considered because is not present in this scenario
        - velocity is not included (since in the original observaion an agent doesn't know other agents' velocities)

        Ex: agent 1 for agent 1:    (0, 0, 1, 1)
        Ex: agent 2 for agent 1:    (dx(a1,a2), dy(a1,a2), 1, 0)
        Ex: landmark 1 for agent 1: (dx(a1,l1), dy(a1,l1), 0, 0)

        """
        agents_pos = world.agents_infos[:, :2]  # position(2) + velocity(2) + size(1)
        agents_vel = world.agents_infos[:, 2:4]  # position(2) + velocity(2) + size(1)
        preys_pos = world.preys_infos[:, :2]  # position(2) + velocity(2) + size(1)
        # 1. Build observation of self and environment's attribution
        # self_label = np.tile(np.array([[1., 1.]]), (world.num_agents, 1))  # (n,2)
        self_info = np.concatenate([agents_vel, agents_pos], axis=-1)

        # 2. Build observation of other agents and preys
        expand_pos_1 = np.expand_dims(agents_pos, axis=1)  # (n_agents, 2) => (n_agents, 1, 2)
        expand_pos_2 = np.expand_dims(agents_pos, axis=0)  # (n_agents, 2) => (1, n_agents, 2)
        expand_pos_3 = np.expand_dims(preys_pos, axis=0)  # (n_preys, 2) => (1, n_preys, 2)

        # 2.1. Compute relative position
        agent_diff_agent = expand_pos_2 - expand_pos_1  # (n_agents, n_agents, 2) (other agents - agent)
        agent_diff_landmark = expand_pos_3 - expand_pos_1  # (n_preys, n_preys, 2) (preys - agent)

        # 2.2. Observe other agents and preys in multi-agent environments
        obs_agents = agent_diff_agent  # (n_agents, n_agents, 2)
        obs_preys = agent_diff_landmark  # (n_agents, n_preys, 2)
        obs_agents[np.eye(world.num_agents) == 1] = np.array([0., 0.])

        # label (1,0) with agents and (0,0) with preys
        agent_label = np.tile(np.array([[[1., 0.]]]), (world.num_agents, world.num_agents, 1))  # (n,n,1)
        agent_label[np.eye(world.num_agents) == 1] = np.array([1., 1.])
        landmark_label = np.tile(np.array([[[0., 0.]]]), (world.num_agents, world.num_preys, 1))  # (n,m,1)

        obs_agents = np.concatenate([obs_agents, agent_label], axis=-1)
        obs_preys = np.concatenate([obs_preys, landmark_label], axis=-1)

        # flatten
        obs_agents_flatten = obs_agents.reshape(world.num_agents, -1)
        obs_preys_flatten = obs_preys.reshape(world.num_agents, -1)

        obs_n = np.concatenate((obs_agents_flatten, obs_preys_flatten), axis=-1)
        world.obs_n = obs_n
        return obs_n

    def entity_state(self, world):

        """
        Entity approach for transformers including velocity

        *5 features for each entity: (pos_x, pos_y, vel_x, vel_y, is_agent)*
        - agent features: (x, y, vel_x, vel_y, 1)
        - landmark features: (x, y, vel_x, vel_y, 0)

        In this case the absolute position of the agents is considered,
        and the velocity is also included. All these informations are present
        alreay in the original state vector, because are included in the agents
        observations which are concatenated together.
        """
        agents_pos = world.agents_infos[:, :2]  # position(2) + velocity(2) + size(1)
        agents_vel = world.agents_infos[:, 2:4]  # position(2) + velocity(2) + size(1)
        preys_pos = world.preys_infos[:, :2]  # position(2) + velocity(2) + size(1)
        preys_vel = world.preys_infos[:, 2:4]  # position(2) + velocity(2) + size(1)

        # label (1) with agents and (0) with preys
        agent_label = np.tile(np.array([[1.]]), (world.num_agents, 1))  # (n,1)
        landmark_label = np.tile(np.array([[0.]]), (world.num_preys, 1))  # (m,1)

        obs_agents_flatten = np.concatenate([agents_pos, agents_vel, agent_label], axis=-1).reshape(1, -1)
        obs_preys_flatten = np.concatenate([preys_pos, preys_vel, landmark_label], axis=-1).reshape(1, -1)
        state_flatten = np.concatenate([obs_agents_flatten, obs_preys_flatten], axis=-1)
        return state_flatten