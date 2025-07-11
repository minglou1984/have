import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm
from modules.layer.permutation import TreeEncoder

class SetMixer(nn.Module):
    def __init__(self, args, abs=True):
        super(SetMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_allies = args.n_allies
        self.n_enemies = args.n_enemies
        self.embed_dim = args.mixing_embed_dim
        self.input_dim = self.state_dim = self.args.rnn_hidden_dim

        self.abs = abs  # monotonicity constraint
        self.qmix_pos_func = getattr(self.args, "qmix_pos_func", "abs")
        assert self.qmix_pos_func == "abs"
        # hyper w1 b1
        self.hyper_w1 = nn.Sequential(nn.Linear(self.input_dim, args.hypernet_embed),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(args.hypernet_embed, self.embed_dim))
        self.hyper_b1 = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim))

        # hyper w2 b2
        self.hyper_w2 = nn.Sequential(nn.Linear(self.input_dim, args.hypernet_embed),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(args.hypernet_embed, self.embed_dim))
        self.hyper_b2 = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(self.embed_dim, 1))

        input_shape = self._get_input_shape()
        self.TreeEncoder = TreeEncoder(input_shape, self.input_dim, args)
        self.observation_hidden = nn.Sequential(nn.Linear(self.input_dim, args.hypernet_embed),
                                                nn.ReLU(),
                                                nn.Linear(args.hypernet_embed, self.embed_dim))
        self.state_hidden = nn.Sequential(nn.Linear(self.input_dim, args.hypernet_embed),
                                          nn.ReLU(),
                                          nn.Linear(args.hypernet_embed, self.embed_dim))
        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim * 2),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim * 2, 1))

        if getattr(args, "use_orthogonal", False):
            for m in self.modules():
                orthogonal_init_(m)

    def forward(self, qvals, states, obs, act, is_alive):
        # reshape
        b, t, _ = qvals.size()
        # is_alive = is_alive.contiguous().view(-1, self.n_agents, 1)  # b * t, n, 1
        act = th.cat([th.zeros_like(act[:,0:1]), act], dim=1)
        # act = states[..., -self.n_agents * self.args.n_actions:].reshape(b*t, self.n_agents, self.args.n_actions).max(dim=-1, keepdim=True)[1] # b * t, n, -1
        inputs = self._build_inputs(obs, act)
        obs_embeddings = self.TreeEncoder(inputs, id_and_action=True)
        obs_embeddings = obs_embeddings.reshape(-1, self.n_agents,
                                              self.state_dim)  # [batch_size*seq_len, n_agents, state_dim]
        # states = (obs_embeddings * is_alive).mean(dim=-2)  # [batch_size*seq_len, state_dim]
        states = obs_embeddings.mean(dim=-2)  # [batch_size*seq_len, state_dim]

        qvals = qvals.reshape(b * t, self.n_agents, 1)
        # First layer
        w1 = self.hyper_w1(states).view(-1, 1, self.embed_dim)  # b * t, n_agents, emb
        b1 = self.hyper_b1(states).view(-1, 1, self.embed_dim)

        # Second layer
        w2 = self.hyper_w2(states).view(-1, self.embed_dim, 1)  # b * t, emb, 1
        b2 = self.hyper_b2(states).view(-1, 1, 1)

        if self.abs:
            w1 = self.pos_func(w1)
            w2 = self.pos_func(w2)
        # print(w1.mean(), w1.var())
        # print(w2.mean(), w2.var())

        # Forward
        hidden = F.elu(th.matmul(qvals, w1) + b1)  # b * t, n, emb
        y = th.matmul(hidden, w2) + b2  # b * t, n, 1

        state_hidden = self.state_hidden(states)  # [batch_size*seq_len, embed_dim]
        observation_hidden = self.observation_hidden(obs_embeddings)  # [batch_size*seq_len, n_agents, embed_dim]

        state_hidden = state_hidden.unsqueeze(dim=-2)  # [batch_size*seq_len, 1, embed_dim]

        _, __, e = observation_hidden.size()
        state_hidden = state_hidden / (e ** (1 / 4))
        observation_hidden = observation_hidden / (e ** (1 / 4))

        # - get dot product of queries and keys
        dot = th.bmm(state_hidden, observation_hidden.transpose(1, 2)).transpose(1, 2)  # [batch_size*seq_len, 1, n_agents] => [batch_size*seq_len, n_agents, 1]
        # dot as row-wise self-attention probabilities
        # dot[is_alive == 0] = -1e10
        # y = y * is_alive
        k = F.softmax(dot, dim=1)

        v = self.V(states).view(-1, 1, 1)
        # y = y.sum(-2, keepdim=True) + v
        # y = y.mean(-2, keepdim=True) + v
        y = th.bmm(y.transpose(1, 2), k) + v  # q_tot = th.bmm(q.transpose(1, 2), k) + v
        # q_tot = th.bmm(agent_qs.transpose(1, 2), k) + v # q_tot = th.bmm(q.transpose(1, 2), k) + v # No
        # q_tot = q.mean(dim=1, keepdim=True) + v # q_tot = th.bmm(q.transpose(1, 2), k) + v # Yes

        # q_tot = th.bmm(q.transpose(1, 2), k) / is_alive.sum(1, keepdim=True).clamp(1, 8) + v

        return y.view(b, t, -1)

    def pos_func(self, x):
        if self.qmix_pos_func == "softplus":
            return th.nn.Softplus(beta=self.args.qmix_pos_func_beta)(x)
        elif self.qmix_pos_func == "quadratic":
            return 0.5 * x ** 2
        else:
            return th.abs(x)

    # Add new func
    def _get_obs_component_dim(self):
        move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim = self.args.obs_component  # [4, (6, 5), (4, 5), 1]
        enemy_feats_dim_flatten = np.prod(enemy_feats_dim)
        ally_feats_dim_flatten = np.prod(ally_feats_dim)
        return (move_feats_dim, enemy_feats_dim_flatten, ally_feats_dim_flatten, own_feats_dim), (
            enemy_feats_dim, ally_feats_dim)

    def _build_inputs(self, obs, act):
        bs, max_t = obs.shape[:2]
        obs_component_dim, _ = self._get_obs_component_dim()
        move_feats, enemy_feats, ally_feats, own_feats = th.split(obs, obs_component_dim, dim=-1)
        enemy_feats = enemy_feats.reshape(bs * max_t * self.n_agents * self.n_enemies,
                                              -1)  # [bs * max_t * n_agents * n_enemies, fea_dim]
        ally_feats = ally_feats.reshape(bs * max_t * self.n_agents * self.n_allies,
                                            -1)  # [bs * max_t * n_agents * n_allies, a_fea_dim]
        # merge move features and own features to simplify computation.
        context_feats = [move_feats, own_feats]  # [batch, max_t, agent_num, own_dim]
        own_context = th.cat(context_feats, dim=-1).reshape(bs * max_t * self.n_agents, -1)  # [bs * max_t * n_agents, own_dim]

        embedding_indices = []
        if self.args.obs_agent_id:
            # agent-id indices, [bs, n_agents]
            embedding_indices.append(th.arange(self.n_agents, device=obs.device).unsqueeze(0).expand(bs * max_t, -1))
        if self.args.obs_last_action:
            # action-id indices, [bs, n_agents]
            embedding_indices.append(act.reshape(bs * max_t, -1))
        return bs * max_t, own_context, enemy_feats, ally_feats, embedding_indices

    def _get_input_shape(self):
        move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim = self.args.obs_component
        own_context_dim = move_feats_dim + own_feats_dim
        return own_context_dim, enemy_feats_dim, ally_feats_dim

