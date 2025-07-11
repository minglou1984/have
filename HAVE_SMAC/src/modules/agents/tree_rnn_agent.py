import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from modules.layer.permutation import Tree_and_Entity_oriented_action_stream, Merger


class Tree_RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(Tree_RNNAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_allies = args.n_allies
        self.n_enemies = args.n_enemies
        self.n_actions = args.n_actions
        self.n_heads = args.hpn_head_num
        self.rnn_hidden_dim = args.rnn_hidden_dim

        # [4 + 1, (6, 5), (4, 5)]
        self.own_feats_dim, self.enemy_feats_dim, self.ally_feats_dim = input_shape
        self.enemy_feats_dim = self.enemy_feats_dim[-1]  # [n_enemies, feat_dim]
        self.ally_feats_dim = self.ally_feats_dim[-1]  # [n_allies, feat_dim]

        self.obs_embedding_and_entity_specific_weights = Tree_and_Entity_oriented_action_stream(input_shape, args)

        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2_basic_actions_stream = nn.Linear(self.rnn_hidden_dim,
                                            args.output_normal_actions)  # (no_op, stop, up, down, right, left)
        self.unify_output_heads = Merger(self.n_heads, 1)
        if self.args.map_type == "MMM":
            assert self.n_enemies >= self.n_agents, "For MMM map, for the reason that the 'attack' and 'rescue' use the same ids in SMAC, n_enemies must >= n_agents"
            self.unify_output_heads_rescue = Merger(self.n_heads, 1)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc2_basic_actions_stream.weight.new(1, self.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # [bs, n_agents, mv_fea_dim], [bs * n_agents * n_enemies, enemy_fea_dim], [bs * n_agents * n_allies, ally_fea_dim], [bs, n_agents, own_fea_dim]
        bs, own_feats_t, enemy_feats_t, ally_feats_t, embedding_indices = inputs

        embedding, hyper_enemy_out, hyper_ally_out = self.obs_embedding_and_entity_specific_weights(inputs)
        x = F.relu(embedding, inplace=True)
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        hh = self.rnn(x, h_in)  # [bs * n_agents, rnn_hidden_dim]

        # Basic Action Stream: Q-values of basic actions
        q_normal = self.fc2_basic_actions_stream(hh).view(bs, self.n_agents, -1)  # [bs, n_agents, 6]

        # Entity-Oriented Action Stream: Q-values of attack actions: [bs * n_agents * n_enemies, rnn_hidden_dim * n_heads]
        fc2_w_attack = hyper_enemy_out[:, -(self.rnn_hidden_dim + 1) * self.n_heads: -self.n_heads].reshape(
            bs * self.n_agents, self.n_enemies, self.rnn_hidden_dim, self.n_heads
        ).transpose(1, 2).reshape(  # -> [bs * n_agents, rnn_hidden_dim, n_enemies, n_heads]
            bs * self.n_agents, self.rnn_hidden_dim, self.n_enemies * self.n_heads
        )  # [bs * n_agents, rnn_hidden_dim, n_enemies * heads]
        fc2_b_attack = hyper_enemy_out[:, -self.n_heads:].reshape(bs * self.n_agents, self.n_enemies * self.n_heads)

        # [bs*n_agents, 1, rnn_hidden_dim] * [bs*n_agents, rnn_hidden_dim, n_enemies*head] -> [bs*n_agents, 1, n_enemies*head]
        q_attacks = (th.matmul(hh.unsqueeze(1), fc2_w_attack).squeeze(1) + fc2_b_attack).view(
            bs * self.n_agents * self.n_enemies, self.n_heads, 1
        )  # [bs * n_agents, n_enemies*head] -> [bs * n_agents * n_enemies, head, 1]

        # Merge multiple heads into one.
        q_attack = self.unify_output_heads(q_attacks).view(  # [bs * n_agents * n_enemies, 1]
            bs, self.n_agents, self.n_enemies
        )  # [bs, n_agents, n_enemies]

        # %%%%%%%%%%%%%%% 'rescue' actions for map_type == "MMM" %%%%%%%%%%%%%%%
        if self.args.map_type == "MMM":
            fc2_w_rescue = hyper_ally_out[:, -(self.rnn_hidden_dim + 1) * self.n_heads: -self.n_heads].reshape(
                bs * self.n_agents, self.n_allies, self.rnn_hidden_dim, self.n_heads
            ).transpose(1, 2).reshape(  # -> [bs * n_agents, rnn_hidden_dim, n_allies, n_heads]
                bs * self.n_agents, self.rnn_hidden_dim, self.n_allies * self.n_heads
            )  # [bs * n_agents, rnn_hidden_dim, n_allies * heads]
            fc2_b_rescue = hyper_ally_out[:, -self.n_heads:].reshape(bs * self.n_agents, self.n_allies * self.n_heads)
            # [bs*n_agents, 1, rnn_hidden_dim] * [bs*n_agents, rnn_hidden_dim, n_allies*head] -> [bs*n_agents, 1, n_allies*head]
            q_rescues = (th.matmul(hh.unsqueeze(1), fc2_w_rescue).squeeze(1) + fc2_b_rescue).view(
                bs * self.n_agents * self.n_allies, self.n_heads, 1
            )  # [bs * n_agents, n_allies*head] -> [bs * n_agents * n_allies, head, 1]
            # Merge multiple heads into one.
            q_rescue = self.unify_output_heads_rescue(q_rescues).view(  # [bs * n_agents * n_allies, 1]
                bs, self.n_agents, self.n_allies
            )  # [bs, n_agents, n_allies]

            # For the reason that medivac is the last indexed agent, so the rescue action idx -> [0, n_allies-1]
            right_padding = th.ones_like(q_attack[:, -1:, self.n_allies:], requires_grad=False) * (-9999999)
            modified_q_attack_of_medivac = th.cat([q_rescue[:, -1:, :], right_padding], dim=-1)
            q_attack = th.cat([q_attack[:, :-1], modified_q_attack_of_medivac], dim=1)

        # Concat 2 types of Q-values
        q = th.cat((q_normal, q_attack), dim=-1)  # [bs, n_agents, 6 + n_enemies]
        return q.view(bs, self.n_agents, -1), hh.view(bs, self.n_agents, -1)  # [bs, n_agents, 6 + n_enemies]
