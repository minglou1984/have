import numpy as np
from ..core import World, Agent, Landmark
from ..scenario import BaseScenario
import random

# Fully Observation: Observe all landmarks and agents in environments
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
        world.dim_c = 2
        world.num_agents = num_agents
        world.num_landmarks = num_landmarks
        world.num_agents_obs = num_agents_obs
        world.num_landmarks_obs = num_landmarks_obs

        world.collaborative = True
        world.discrete_action = True

        world.agents_infos = np.zeros((num_agents, 6)) # position(2) + velocity(2) + size(1) + observe capacity (1, MAX=2*1.414=2.828)
        world.landmarks_infos = np.zeros((num_landmarks, 5)) # position(2) + velocity(2) + size(1)
        self.entity_obs_feats = 4
        self.entity_state_feats = 5
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            world.agents_infos[i, 4] = 0.04 # size
            agent.size = world.agents_infos[i, 4]
            world.agents_infos[i, 5] = 2.0 # observe distance, only for partial observable
            # agent.accel = 5.0
            # agent.accel = 3.0
            # agent.max_speed = 5.0

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            world.landmarks_infos[i, 4] = 0.08 # size
            landmark.size = world.landmarks_infos[i, 4]
        # Landmark Size: 0.07
        # Agent Size: 0.035
        # Accel, Predator:Prey: 3.0;4.0
        # Max Speed, Predator:Prey: 1;1.3
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for i, agent in enumerate(world.agents):
            world.agents_infos[i, :2] = world.np_random.uniform(-world.range_p, +world.range_p, world.dim_p)
            world.agents_infos[i, 2:4] = np.zeros(world.dim_p)
            agent.state.p_pos = world.agents_infos[i, :2]
            agent.state.p_vel = world.agents_infos[i, 2:4]
            agent.state.c = np.zeros(world.dim_c)

        for i, landmark in enumerate(world.landmarks):
            world.landmarks_infos[i, :2] = world.np_random.uniform(-world.range_p, +world.range_p, world.dim_p)
            world.landmarks_infos[i, 2:4] = np.zeros(world.dim_p)
            landmark.state.p_pos = world.landmarks_infos[i, :2]
            landmark.state.p_vel = world.landmarks_infos[i, 2:4]
            # avoid overlap
            for j in range(5):
                if i == 0: break
                agent_diff_other = world.landmarks_infos[i, :2] - world.landmarks_infos[:i, :2]
                agent_dist_other = np.linalg.norm(agent_diff_other, axis=-1)
                dist_mask = agent_dist_other < landmark.size * 3
                if sum(dist_mask) < 1:
                    break
                else:
                    world.landmarks_infos[i, :2] = world.np_random.uniform(-world.range_p, +world.range_p,
                                                                           world.dim_p)
                    landmark.state.p_pos = world.landmarks_infos[i, :2]


    # original
    def benchmark_data_v1(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0

        for l in world.landmarks:
            occupied_dist = agent.size + l.size
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < occupied_dist:
                occupied_landmarks += 1
        if agent.collide:
            for i, a in enumerate(world.agents):
                for j in range(i,world.num_agents):
                    b = world.agents[j]
                    if a is b: continue
                    if self.is_collision(a, b):
                        '''collide reward -1'''
                        collisions += 1
                        rew -= 1

        info = {}
        info['rew'] = rew
        info['collisions'] = collisions
        info['min_dists'] = min_dists
        info['occ_land'] = occupied_landmarks
        info['occ_land_rate'] = occupied_landmarks / world.num_landmarks

        return info

    def benchmark_data(self, agent, world):
        rew_n = 0
        collisions_n = 0
        occupied_landmarks_n = 0

        agents_pos = world.agents_infos[:, :2] # position(2) + velocity(2) + size(1) + observe capacity(1)
        landmarks_pos = world.landmarks_infos[:, :2] # position(2) + velocity(2) + size(1)
        agents_size = world.agents_infos[:, 4] # (n_agents,)
        landmarks_size = world.landmarks_infos[:, 4] # (n_landmarks,)

        expand_pos_1 = np.expand_dims(landmarks_pos, axis=1) # (n_landmarks, 2) => (n_landmarks, 1, 2)
        expand_pos_2 = np.expand_dims(agents_pos, axis=0) # (n_agents, 2) => (1, n_agents, 2)
        expand_pos_3 = np.expand_dims(agents_pos, axis=1) # (n_agents, 2) => (n_agents, 1, 2)

        # compute relative distance
        landmark_diff_agent = expand_pos_1 - expand_pos_2 # (n_landmarks, n_agents, 2) (landmark - other agents)
        agent_diff_agent = expand_pos_2 - expand_pos_3 # (n_agents, n_agents, 2) (other agents - agent)
        landmark_dist_agent = np.linalg.norm(landmark_diff_agent, axis=-1)  # (n_landmarks, n_agents)
        agent_dist_agent = np.linalg.norm(agent_diff_agent, axis=-1)  # (n_agents, n_agents)

        # occupied distance
        occupied_dist = np.expand_dims(agents_size, axis=0) + np.expand_dims(landmarks_size, axis=1) # (n_landmarks, n_agents)
        # collision distance
        collision_dist = np.expand_dims(agents_size, axis=1) + np.expand_dims(agents_size,
                                                                             axis=0)  # (n_agents, n_agents)

        collision_mask = agent_dist_agent<collision_dist

        collisions_n += (collision_mask.sum()-world.num_agents) / 2
        min_dists = np.min(landmark_dist_agent, axis=-1) # (n_landmarks,)
        occupied_landmarks_n += (np.sum(landmark_dist_agent < occupied_dist, axis=-1) > 0).sum()

        min_dists_n = min_dists.sum()
        rew_n -= min_dists_n + collisions_n

        info = {}
        info['rew'] = rew_n
        info['collisions'] = collisions_n
        info['min_dists'] = min_dists_n
        info['occ_land'] = occupied_landmarks_n
        info['occ_land_rate'] = occupied_landmarks_n / world.num_landmarks

        return info

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        collision_dist = agent1.size + agent2.size
        return True if dist < collision_dist else False

    # original
    def reward_v1(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        # global reward
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
        # collisions penalty
        if agent.collide:
            for i, a in enumerate(world.agents):
                for j in range(i,world.num_agents):
                    b = world.agents[j]
                    if a is b: continue
                    if self.is_collision(a, b):
                        '''collide reward -1'''
                        rew -= 1
        return rew

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew_n = 0
        # global reward
        collisions_n = 0

        agents_pos = world.agents_infos[:, :2] # position(2) + velocity(2) + size(1) + observe capacity(1)
        landmarks_pos = world.landmarks_infos[:, :2] # position(2) + velocity(2) + size(1)
        agents_size = world.agents_infos[:, 4] # (n_agents,)

        expand_pos_1 = np.expand_dims(landmarks_pos, axis=1) # (n_landmarks, 2) => (n_landmarks, 1, 2)
        expand_pos_2 = np.expand_dims(agents_pos, axis=0) # (n_agents, 2) => (1, n_agents, 2)
        expand_pos_3 = np.expand_dims(agents_pos, axis=1) # (n_agents, 2) => (n_agents, 1, 2)

        # compute relative distance
        landmark_diff_agent = expand_pos_1 - expand_pos_2  # (n_landmarks, n_agents, 2)
        agent_diff_agent = expand_pos_2 - expand_pos_3 # (n_agents, n_agents, 2)
        landmark_dist_agent = np.linalg.norm(landmark_diff_agent, axis=-1)  # (n_landmarks, n_agents)
        agent_dist_agent = np.linalg.norm(agent_diff_agent, axis=-1)  # (n_agents, n_agents)

        collision_dist = np.expand_dims(agents_size, axis=1) + np.expand_dims(agents_size,
                                                                              axis=0)  # (n_agents, n_agents)

        collision_mask = agent_dist_agent < collision_dist

        collisions_n += (collision_mask.sum() - world.num_agents) / 2
        min_dists = np.min(landmark_dist_agent, axis=-1)
        min_dists_n = min_dists.sum()

        rew_n -= min_dists_n + collisions_n
        return rew_n

    def observation(self, agent, world):
        for i, other in enumerate(world.agents):
            if i==0 and other is agent:
                self.generate_observation(world)
            if agent is other:
                return world.obs_n[i]

    # original
    def observation_v1(self, agent, world):
        # landmark part
        landmark_pos = []
        for entity in world.landmarks:  # world.entities:
            landmark_pos.append(entity.state.p_pos - agent.state.p_pos)

        # agent part
        other_agent_pos = []
        for other in world.agents:
            if other is agent: continue
            other_agent_pos.append(other.state.p_pos - agent.state.p_pos)

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_agent_pos + landmark_pos)

    def generate_observation(self, world):
        agents_pos = world.agents_infos[:, :2] # position(2) + velocity(2) + size(1)
        agents_vel = world.agents_infos[:, 2:4] # position(2) + velocity(2) + size(1)
        landmarks_pos = world.landmarks_infos[:, :2] # position(2) + velocity(2) + size(1)
        expand_pos_1 = np.expand_dims(agents_pos, axis=1) # (n_agents, 2) => (n_agents, 1, 2)
        expand_pos_2 = np.expand_dims(agents_pos, axis=0) # (n_agents, 2) => (1, n_agents, 2)
        expand_pos_3 = np.expand_dims(landmarks_pos, axis=0) # (n_landmarks, 2) => (1, n_landmarks, 2)

        # compute relative position
        agent_diff_agent = expand_pos_2 - expand_pos_1 # (n_agents, n_agents, 2) (other agents - agent)
        agent_diff_landmark = expand_pos_3 - expand_pos_1 # (n_landmarks, n_landmarks, 2) (landmarks - agent)

        # 1. observe other agents and landmarks in specific nearest number
        # 1.1. compute relative distance
        # agent_dist_agent = np.linalg.norm(agent_diff_agent, axis=-1, keepdims=True)  # (n_agents, n_agents, 1)
        # agent_dist_landmark = np.linalg.norm(agent_diff_landmark, axis=-1, keepdims=True)  # (n_agents, n_landmarks, 1)
        # 1.2. sort by relative distance
        # sorted_agent_indices = np.argsort(agent_dist_agent, axis=1)[:, 1:] # (n_agents, n_agents-1, 1), exclude self
        # sorted_landmark_indices = np.argsort(agent_dist_landmark, axis=1) # (n_agents, n_landmarks, 1)
        # 1.3. select nearest the agents and landmarks
        # obs_agents = np.take_along_axis(agent_diff_agent, sorted_agent_indices, axis=1)[:, :world.num_agents_obs]
        # obs_landmarks = np.take_along_axis(agent_diff_landmark, sorted_landmark_indices, axis=1)[:, :world.num_landmarks_obs]

        # 2.observe other agents and landmarks in multi-agent environments
        obs_agents = agent_diff_agent # (n_agents, n_agents, 2)
        obs_landmarks = agent_diff_landmark # (n_agents, n_landmarks, 2)
        obs_agents = obs_agents[np.eye(world.num_agents) == 0].reshape(world.num_agents, -1, 2) # (n_agents, n_agents-1, 2)

        # flatten
        obs_agents_flatten = obs_agents.reshape(world.num_agents, -1)
        obs_landmarks_flatten = obs_landmarks.reshape(world.num_agents, -1)

        obs_n = np.concatenate((agents_vel, agents_pos, obs_agents_flatten, obs_landmarks_flatten), axis=-1)
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
        landmarks_pos = world.landmarks_infos[:, :2]  # position(2) + velocity(2) + size(1)
        landmarks_vel = world.landmarks_infos[:, 2:4]  # position(2) + velocity(2) + size(1)

        # label (1) with agents and (0) with landmarks
        agent_label = np.tile(np.array([[1.]]), (world.num_agents, 1))  # (n,1)
        landmark_label = np.tile(np.array([[0.]]), (world.num_landmarks, 1))  # (m,1)

        obs_agents_flatten = np.concatenate([agents_pos, agents_vel, agent_label], axis=-1).reshape(1, -1)
        obs_landmarks_flatten = np.concatenate([landmarks_pos, landmarks_vel, landmark_label], axis=-1).reshape(1, -1)
        state_flatten = np.concatenate([obs_agents_flatten, obs_landmarks_flatten], axis=-1)
        return state_flatten

        # entity_state_feats = 5
        # num_entities = world.num_agents + world.num_landmarks
        #
        # feats = np.zeros(entity_state_feats * num_entities)
        #
        # # agents features
        # i = 0
        # for a in world.agents:
        #     pos = a.state.p_pos
        #     vel = a.state.p_vel
        #     feats[i:i + entity_state_feats] = [pos[0], pos[1], vel[0], vel[1], 1.]
        #     i += entity_state_feats
        #
        # # landmarks features
        # for landmark in world.landmarks:
        #     pos = landmark.state.p_pos
        #     vel = landmark.state.p_vel  # the velocity in this case is just 0
        #     feats[i:i + entity_state_feats] = [pos[0], pos[1], vel[0], vel[1], 0.]
        #     i += entity_state_feats
        #
        # return feats

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
        landmarks_pos = world.landmarks_infos[:, :2]  # position(2) + velocity(2) + size(1)
        # 1. Build observation of self and environment's attribution
        # self_label = np.tile(np.array([[1., 1.]]), (world.num_agents, 1))  # (n,2)
        self_info = np.concatenate([agents_vel, agents_pos], axis=-1)

        # 2. Build observation of other agents and landmarks
        expand_pos_1 = np.expand_dims(agents_pos, axis=1)  # (n_agents, 2) => (n_agents, 1, 2)
        expand_pos_2 = np.expand_dims(agents_pos, axis=0)  # (n_agents, 2) => (1, n_agents, 2)
        expand_pos_3 = np.expand_dims(landmarks_pos, axis=0)  # (n_landmarks, 2) => (1, n_landmarks, 2)

        # 2.1. Compute relative position
        agent_diff_agent = expand_pos_2 - expand_pos_1  # (n_agents, n_agents, 2) (other agents - agent)
        agent_diff_landmark = expand_pos_3 - expand_pos_1  # (n_landmarks, n_landmarks, 2) (landmarks - agent)

        # 2.2. Observe other agents and landmarks in multi-agent environments
        obs_agents = agent_diff_agent  # (n_agents, n_agents, 2)
        obs_landmarks = agent_diff_landmark  # (n_agents, n_landmarks, 2)
        obs_agents[np.eye(world.num_agents) == 1] = np.array([0., 0.])

        # label (1,0) with agents and (0,0) with landmarks
        agent_label = np.tile(np.array([[[1., 0.]]]), (world.num_agents, world.num_agents, 1))  # (n,n,1)
        agent_label[np.eye(world.num_agents) == 1] = np.array([1., 1.])
        landmark_label = np.tile(np.array([[[0., 0.]]]), (world.num_agents, world.num_landmarks, 1))  # (n,m,1)

        obs_agents = np.concatenate([obs_agents, agent_label], axis=-1)
        obs_landmarks = np.concatenate([obs_landmarks, landmark_label], axis=-1)

        # flatten
        obs_agents_flatten = obs_agents.reshape(world.num_agents, -1)
        obs_landmarks_flatten = obs_landmarks.reshape(world.num_agents, -1)

        obs_n = np.concatenate((obs_agents_flatten, obs_landmarks_flatten), axis=-1)
        world.obs_n = obs_n
        return obs_n







        # num_entities = world.num_agents + world.num_landmarks
        # entity_obs_feats = 4
        # feats = np.zeros(entity_obs_feats * num_entities)
        #
        # # agent features
        # pos_a = agent.state.p_pos
        # i = 0
        # for a in world.agents:
        #     if a is agent:
        #         # vel = agent.state.p_vel
        #         feats[i:i + entity_obs_feats] = [0., 0., 1., 1.]
        #     else:
        #         pos = a.state.p_pos - pos_a
        #         feats[i:i + entity_obs_feats] = [pos[0], pos[1], 1., 0.]
        #     i += entity_obs_feats
        #
        # # landmarks features
        # for j, landmark in enumerate(world.landmarks):
        #     pos = landmark.state.p_pos - pos_a
        #     feats[i:i + entity_obs_feats] = [pos[0], pos[1], 0., 0.]
        #     i += entity_obs_feats
        #
        # return feats