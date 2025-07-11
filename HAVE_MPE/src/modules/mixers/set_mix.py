import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from modules.layer.permutation import TreeEncoder

class SetMixer(nn.Module):
    def __init__(self, args):
        super(SetMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_enemies = args.n_enemies
        self.n_allies = args.n_allies

        self.state_dim = args.hpn_qmix_hidden_dim
        self.observation_dim = self.state_dim
        self.embed_dim = args.mixing_embed_dim
        hypernet_embed = self.args.hypernet_embed

        input_shape = self._get_input_shape()
        self.permutation_invariant_input = TreeEncoder(input_shape, args.hpn_qmix_hidden_dim, args)

        self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                       nn.ReLU(),
                                       nn.Linear(hypernet_embed, self.embed_dim))
        self.hyper_w_q = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                       nn.ReLU(),
                                       nn.Linear(hypernet_embed, self.embed_dim))

        self.observation_hidden = nn.Sequential(nn.Linear(self.observation_dim, hypernet_embed),
                                                nn.ReLU(),
                                                nn.Linear(hypernet_embed, self.embed_dim))
        self.state_hidden = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                          nn.ReLU(),
                                          nn.Linear(hypernet_embed, self.embed_dim))

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
        self.hyper_b_q = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim * 2),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim * 2, 1))
        self.k = None

    def forward(self, agent_qs, n_observations):
        bs = agent_qs.size(0)  # [batch_size, seq_len, n_agents]
        n_observations = n_observations.reshape(*n_observations.shape[:2], self.n_agents, -1)  # [batch_size, seq_len, n_agents, obs_shape]
        bs, max_t, own_inputs, ally_feats, enemy_feats = self._build_inputs(n_observations)
        input_states = (bs * max_t, own_inputs, ally_feats, enemy_feats)
        n_observations = self.permutation_invariant_input(input_states) # [batch_size*seq_len*n_agents, hpn_qmix_hidden_dim]

        observations= n_observations.reshape(-1, self.n_agents, self.state_dim) # [batch_size*seq_len, n_agents, state_dim]
        states = observations.mean(dim=-2) # [batch_size*seq_len, state_dim]

        agent_qs = agent_qs.view(-1, self.n_agents, 1) # [batch_size * seq_len, n_agents , 1]
        # First layer
        w1 = th.abs(self.hyper_w_1(states)) # state_dim => embed_dim , W(bs,embed_dim)
        b1 = self.hyper_b_1(states) # B(bs, state_dim)
        w1 = w1.view(-1, 1, self.embed_dim) # W(bs, 1, embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim) # B(bs, 1, embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1) # (batch_size * seq_len, n_agents , embed_dim)

        # calculate Q and K separately
        w_q = th.abs(self.hyper_w_q(states)).view(-1, self.embed_dim, 1) # (batch_size * seq_len, embed_dim , 1)
        b_q = self.hyper_b_q(states).view(-1, 1, 1)
        q = th.bmm(hidden, w_q) + b_q # (batch_size * seq_len, n_agents , 1)

        state_hidden = self.state_hidden(states) # [batch_size*seq_len, embed_dim]
        observation_hidden = self.observation_hidden(observations) # [batch_size*seq_len, n_agents, embed_dim]

        state_hidden = state_hidden.unsqueeze(dim=-2) # [batch_size*seq_len, 1, embed_dim]

        _, __, e = observation_hidden.size()
        state_hidden = state_hidden / (e ** (1 / 4))
        observation_hidden = observation_hidden / (e ** (1 / 4))

        # - get dot product of queries and keys
        dot = th.bmm(state_hidden, observation_hidden.transpose(1, 2)) # [batch_size*seq_len, 1, n_agents]
        # dot as row-wise self-attention probabilities
        k = F.softmax(dot, dim=2).transpose(1, 2) # [batch_size*seq_len, n_agents, 1]

        v = self.V(states).view(-1, 1, 1)
        q_tot = th.bmm(q.transpose(1, 2), k) + v # q_tot = th.bmm(q.transpose(1, 2), k) + v
        self.q = q
        q_tot = q_tot.view(bs, -1, 1)

        return q_tot

    def _get_input_shape(self):
        own_feats_dim, ally_feats_dim, enemy_feats_dim = self.args.obs_component
        own_input_shape = own_feats_dim
        ally_input_shape = copy.deepcopy(ally_feats_dim)
        enemy_input_shape = copy.deepcopy(enemy_feats_dim)

        return own_input_shape, ally_input_shape, enemy_input_shape

    def _get_obs_component_dim(self):
        own_feats_dim, ally_feats_dim, enemy_feats_dim  = self.args.obs_component  # [(5, 2), (4, 2), 4]
        ally_feats_dim_flatten = np.prod(ally_feats_dim)
        enemy_feats_dim_flatten = np.prod(enemy_feats_dim)
        return (own_feats_dim, ally_feats_dim_flatten, enemy_feats_dim_flatten), (
            enemy_feats_dim, ally_feats_dim)

    def _build_inputs(self, state):
        bs, max_t = state.shape[:2]
        obs_component_dim, _ = self._get_obs_component_dim()
        raw_obs = state
        own_feats, ally_feats, enemy_feats = th.split(raw_obs, obs_component_dim, dim=-1)

        ally_feats = ally_feats.reshape(bs * max_t * self.n_agents * self.n_allies,
                                            -1)  # [bs * n_agents * n_allies, a_fea_dim]
        enemy_feats = enemy_feats.reshape(bs *max_t* self.n_agents * self.n_enemies,
                                              -1)  # [bs * n_agents * n_enemies, fea_dim]

        own_inputs = own_feats.reshape(bs * max_t * self.n_agents, -1)  # [bs * seq_len * n_agents, fea_dim]

        return bs, max_t, own_inputs, ally_feats, enemy_feats

class DyanmicMixer_Independent(nn.Module):
    def __init__(self, args):
        super(DyanmicMixer_Independent, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_enemies = args.n_enemies
        self.n_allies = args.n_allies

        input_shape = self._get_input_shape()
        self.DerNet = DerNet(input_shape, args.hpn_qmix_hidden_dim, args)

        self.state_dim = args.hpn_qmix_hidden_dim
        self.embed_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                               nn.ReLU(),
                                               nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0) # [batch_size, seq_len, n_agents]
        states = states.reshape(*states.shape[:2], self.n_agents, -1) # [batch_size, seq_len, n_agents, obs_shape]
        bs, max_t, own_inputs, ally_feats, enemy_feats = self._build_inputs(states)
        input_states = (bs * max_t, own_inputs, ally_feats, enemy_feats)
        states = self.DerNet(input_states) # [batch_size*seq_len*n_agents, hpn_qmix_hidden_dim]

        states = states.reshape(-1, self.n_agents, self.state_dim).mean(dim=-2) # [batch_size*seq_len, state_dim]

        agent_qs = agent_qs.view(-1, 1, self.n_agents) # [batch_size*seq_len, 1, n_agents]
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1) # [batch_size, seq_len, 1]
        return q_tot

    def _get_input_shape(self):
        own_feats_dim, ally_feats_dim, enemy_feats_dim = self.args.obs_component
        own_input_shape = own_feats_dim
        ally_input_shape = copy.deepcopy(ally_feats_dim)
        enemy_input_shape = copy.deepcopy(enemy_feats_dim)

        return own_input_shape, ally_input_shape, enemy_input_shape

    def _get_obs_component_dim(self):
        own_feats_dim, ally_feats_dim, enemy_feats_dim  = self.args.obs_component  # [(5, 2), (4, 2), 4]
        ally_feats_dim_flatten = np.prod(ally_feats_dim)
        enemy_feats_dim_flatten = np.prod(enemy_feats_dim)
        return (own_feats_dim, ally_feats_dim_flatten, enemy_feats_dim_flatten), (
            enemy_feats_dim, ally_feats_dim)

    def _build_inputs(self, state):
        bs, max_t = state.shape[:2]
        obs_component_dim, _ = self._get_obs_component_dim()
        raw_obs = state
        own_feats, ally_feats, enemy_feats = th.split(raw_obs, obs_component_dim, dim=-1)

        ally_feats = ally_feats.reshape(bs * max_t * self.n_agents * self.n_allies,
                                            -1)  # [bs * n_agents * n_allies, a_fea_dim]
        enemy_feats = enemy_feats.reshape(bs *max_t* self.n_agents * self.n_enemies,
                                              -1)  # [bs * n_agents * n_enemies, fea_dim]

        own_inputs = own_feats.reshape(bs * max_t * self.n_agents, -1)  # [bs * seq_len * n_agents, fea_dim]

        return bs, max_t, own_inputs, ally_feats, enemy_feats