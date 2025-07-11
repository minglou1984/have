import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from modules.layer.multihead_attention import MHA, MultiHeadAttention

class TreeEncoder(nn.Module):
    def __init__(self, input_shape, output_shape, args):
        super(TreeEncoder, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_allies = args.n_allies
        self.n_enemies = args.n_enemies
        self.n_actions = args.n_actions
        self.n_heads = args.hpn_head_num # default: 1
        self.output_shape = output_shape  # default: 64

        # [4 + 1, (6, 5), (4, 5)]
        self.own_feats_dim, self.enemy_feats_dim, self.ally_feats_dim = input_shape
        self.enemy_feats_dim = self.enemy_feats_dim[-1]  # [n_enemies, feat_dim]
        self.ally_feats_dim = self.ally_feats_dim[-1]  # [n_allies, feat_dim]

        if self.args.obs_agent_id:
            # embedding table for agent_id
            self.agent_id_embedding = th.nn.Embedding(self.n_agents, output_shape)

        if self.args.obs_last_action:
            # embedding table for action id
            self.action_id_embedding = th.nn.Embedding(self.n_actions, output_shape)

        # Unique Features (do not need hyper net)
        self.fc1_own = nn.Linear(self.own_feats_dim, output_shape, bias=True)  # only one bias is OK

        # Multiple entities (use hyper net to process these features to ensure permutation invariant)
        self.hyper_input_w_enemy = Hypernet(
            input_dim=self.enemy_feats_dim, hidden_dim=args.hpn_hyper_dim,
            main_input_dim=self.enemy_feats_dim, main_output_dim=output_shape,
            activation_func=args.hpn_hyper_activation, n_heads=self.n_heads
        )  # output shape: (enemy_feats_dim * self.output_shape)
        self.hyper_input_w_ally = Hypernet(
            input_dim=self.ally_feats_dim, hidden_dim=args.hpn_hyper_dim,
            main_input_dim=self.ally_feats_dim, main_output_dim=output_shape,
            activation_func=args.hpn_hyper_activation, n_heads=self.n_heads
        )  # output shape: ally_feats_dim * output_shape

        self.attention_enemy = MHA(output_shape, heads=args.dernet_head_num)
        self.attention_ally = MHA(output_shape, heads=args.dernet_head_num)
        self.unify_input_heads = Merger(self.n_heads, output_shape)

    def forward(self, inputs, id_and_action=True):
        # [bs, n_agents, mv_fea_dim], [bs * n_agents * n_enemies, enemy_fea_dim], [bs * n_agents * n_allies, ally_fea_dim], [bs, n_agents, own_fea_dim]
        bs, own_feats_t, enemy_feats_t, ally_feats_t, embedding_indices = inputs

        # (1) Own feature
        embedding_own = self.fc1_own(own_feats_t)  # [bs * n_agents, output_shape]

        # (2) ID embeddings
        if self.args.obs_agent_id and id_and_action:
            agent_indices = embedding_indices[0]
            # [bs * n_agents, output_shape]
            embedding_own = embedding_own + self.agent_id_embedding(agent_indices).view(-1, self.output_shape)
        if self.args.obs_last_action and id_and_action:
            last_action_indices = embedding_indices[-1]
            if last_action_indices is not None:  # t != 0
                # [bs * n_agents, output_shape]
                embedding_own = embedding_own + self.action_id_embedding(last_action_indices).view(
                    -1, self.output_shape)

        # (3) Enemy feature  (enemy_feats_dim * output_shape + output_shape + 1)
        input_w_enemy = self.hyper_input_w_enemy(enemy_feats_t)
        # [bs * n_agents * n_enemies, 1, enemy_fea_dim] * [bs * n_agents * n_enemies, enemy_fea_dim, output_shape] = [bs * n_agents * n_enemies, 1, output_shape]
        # Linear transform:
        embedding_enemies = th.matmul(
            enemy_feats_t.unsqueeze(1),
            input_w_enemy
        ).view(bs * self.n_agents, self.n_enemies, self.n_heads, self.output_shape)  # [bs * n_agents, n_enemies, n_heads, output_shape]

        # Pool: Weight_Sum (Attention)
        embedding_enemies = (embedding_enemies.permute(0, 2, 1, 3)
                             .reshape(bs * self.n_agents * self.n_heads,self.n_enemies, -1))
        # mask_enemies = (th.abs(enemy_feats_t).sum(dim=-1, keepdim=True) != 0).reshape(-1, 1,
        #                                                                               self.n_enemies)  # [bs * n_agents, 1, n_enemies]
        embedding_enemies = self.attention_enemy(
            embedding_own.unsqueeze(dim=1).repeat(self.n_heads, 1, 1),  # K: [bs * n_agents * n_head, 1, hidden_dim]
            embedding_enemies,  # Q: [bs * n_agents * n_head, n_enemies, hidden_dim]
            embedding_enemies,  # V: [bs * n_agents * n_head, n_enemies, hidden_dim]
            # mask_enemies.repeat(self.n_heads, 1, 1)  # MASK: [bs * n_agents * n_head, 1, n_allies]
        ).view(bs * self.n_agents, self.n_heads, -1)  # [bs * n_agents, n_head, hidden_dim]
        # Pool: Sum
        # embedding_enemies = embedding_enemies.sum(dim=1, keepdim=False)  # [bs * n_agents, n_heads, output_shape]

        # (4) Ally features
        input_w_ally = self.hyper_input_w_ally(ally_feats_t)
        # [bs * n_agents * n_allies, 1, ally_fea_dim] * [bs * n_agents * n_allies, ally_fea_dim, n_heads* output_shape] = [bs * n_agents * n_allies, 1, n_heads*output_shape]
        # Linear transform:
        embedding_allies = th.matmul(
            ally_feats_t.unsqueeze(1),
            input_w_ally
        ).view(bs * self.n_agents, self.n_allies, self.n_heads, self.output_shape)  # [bs * n_agents, n_allies, head, output_shape]

        # Pool: Weight_Sum (Attention)
        embedding_allies = (embedding_allies.permute(0, 2, 1, 3)
                             .reshape(bs * self.n_agents * self.n_heads, self.n_allies, -1))
        # mask_allies = (th.abs(ally_feats_t).sum(dim=-1, keepdim=True) != 0).reshape(-1, 1,
        #                                                                             self.n_allies)  # [bs * n_agents, 1, n_allies]
        embedding_allies = self.attention_ally(
            embedding_own.unsqueeze(dim=1).repeat(self.n_heads, 1, 1),  # K: [bs * n_agents * n_head, 1, hidden_dim]
            embedding_allies,  # Q: [bs * n_agents * n_head, n_enemies, hidden_dim]
            embedding_allies,  # V: [bs * n_agents * n_head, n_enemies, hidden_dim]
            # mask_allies.repeat(self.n_heads, 1, 1)  # MASK: [bs * n_agents * n_head, 1, n_allies]
        ).view(bs * self.n_agents, self.n_heads, -1)  # [bs * n_agents, n_head, hidden_dim]

        # Pool: Sum
        # embedding_allies = embedding_allies.sum(dim=1, keepdim=False)  # [bs * n_agents, head, output_shape]

        # Final embedding
        embedding = embedding_own + self.unify_input_heads(
            embedding_enemies + embedding_allies
        )  # [bs * n_agents, head, output_shape]
        return embedding

class Tree_and_Entity_oriented_action_stream(nn.Module):
    def __init__(self, input_shape, args):
        super(Tree_and_Entity_oriented_action_stream, self).__init__()
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

        if self.args.obs_agent_id:
            # embedding table for agent_id
            self.agent_id_embedding = th.nn.Embedding(self.n_agents, self.rnn_hidden_dim)

        if self.args.obs_last_action:
            # embedding table for action id
            self.action_id_embedding = th.nn.Embedding(self.n_actions, self.rnn_hidden_dim)

        # Unique Features (do not need hyper net)
        self.fc1_own = nn.Linear(self.own_feats_dim, self.rnn_hidden_dim, bias=True)  # only one bias is OK

        # Multiple entities (use hyper net to process these features to ensure permutation invariant)
        self.hyper_enemy = nn.Sequential(
            nn.Linear(self.enemy_feats_dim, args.hpn_hyper_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hpn_hyper_dim, ((self.enemy_feats_dim + 1) * self.rnn_hidden_dim + 1) * self.n_heads)
        )  # output shape: (enemy_feats_dim * rnn_hidden_dim + rnn_hidden_dim + 1)

        if self.args.map_type == "MMM":
            assert self.n_enemies >= self.n_agents, "For MMM map, for the reason that the 'attack' and 'rescue' use the same ids in SMAC, n_enemies must >= n_agents"
            self.hyper_ally = nn.Sequential(
                nn.Linear(self.ally_feats_dim, args.hpn_hyper_dim),
                nn.ReLU(inplace=True),
                nn.Linear(args.hpn_hyper_dim, ((self.ally_feats_dim + 1) * self.rnn_hidden_dim + 1) * self.n_heads)
            )  # output shape: ally_feats_dim * rnn_hidden_dim + rnn_hidden_dim + 1, for 'rescue actions'
            # self.unify_output_heads_rescue = Merger(self.n_heads, 1)
        else:
            self.hyper_ally = nn.Sequential(
                nn.Linear(self.ally_feats_dim, args.hpn_hyper_dim),
                nn.ReLU(inplace=True),
                nn.Linear(args.hpn_hyper_dim, self.ally_feats_dim * self.rnn_hidden_dim * self.n_heads)
            )  # output shape: ally_feats_dim * rnn_hidden_dim
        self.attention_enemy = MHA(self.rnn_hidden_dim, heads=args.dernet_head_num)
        self.attention_ally = MHA(self.rnn_hidden_dim, heads=args.dernet_head_num)
        self.unify_input_heads = Merger(self.n_heads, self.rnn_hidden_dim)


    def forward(self, inputs):
        # [bs, n_agents, mv_fea_dim], [bs * n_agents * n_enemies, enemy_fea_dim], [bs * n_agents * n_allies, ally_fea_dim], [bs, n_agents, own_fea_dim]
        bs, own_feats_t, enemy_feats_t, ally_feats_t, embedding_indices = inputs

        # (1) Own feature
        embedding_own = self.fc1_own(own_feats_t)  # [bs * n_agents, rnn_hidden_dim]

        # (2) ID embeddings
        if self.args.obs_agent_id:
            agent_indices = embedding_indices[0]
            # [bs * n_agents, rnn_hidden_dim]
            embedding_own = embedding_own + self.agent_id_embedding(agent_indices).view(-1, self.rnn_hidden_dim)
        if self.args.obs_last_action:
            last_action_indices = embedding_indices[-1]
            if last_action_indices is not None:  # t != 0
                # [bs * n_agents, rnn_hidden_dim]
                embedding_own = embedding_own + self.action_id_embedding(last_action_indices).view(
                    -1, self.rnn_hidden_dim)

        # (3) Enemy feature  (enemy_feats_dim * rnn_hidden_dim + rnn_hidden_dim + 1)
        hyper_enemy_out = self.hyper_enemy(enemy_feats_t)
        fc1_w_enemy = hyper_enemy_out[:, :-(self.rnn_hidden_dim + 1) * self.n_heads].reshape(
            -1, self.enemy_feats_dim, self.rnn_hidden_dim * self.n_heads
        )  # [bs * n_agents * n_enemies, enemy_fea_dim, rnn_hidden_dim]
        # [bs * n_agents * n_enemies, 1, enemy_fea_dim] * [bs * n_agents * n_enemies, enemy_fea_dim, rnn_hidden_dim] = [bs * n_agents * n_enemies, 1, rnn_hidden_dim]
        embedding_enemies = th.matmul(enemy_feats_t.unsqueeze(1), fc1_w_enemy).view(
            bs * self.n_agents, self.n_enemies, self.n_heads, self.rnn_hidden_dim
        )  # [bs * n_agents, n_enemies, n_heads, rnn_hidden_dim]
        # Pool: Weight_Sum (Attention)
        embedding_enemies = (embedding_enemies.permute(0, 2, 1, 3)
                             .reshape(bs * self.n_agents * self.n_heads, self.n_enemies, -1))
        embedding_enemies = self.attention_enemy(
            embedding_own.unsqueeze(dim=1).repeat(self.n_heads, 1, 1),  # K: [bs * n_agents * n_head, 1, hidden_dim]
            embedding_enemies,  # Q: [bs * n_agents * n_head, n_enemies, hidden_dim]
            embedding_enemies,  # V: [bs * n_agents * n_head, n_enemies, hidden_dim]
            # mask_enemies.repeat(self.n_heads, 1, 1)  # MASK: [bs * n_agents * n_head, 1, n_allies]
        ).view(bs * self.n_agents, self.n_heads, -1)  # [bs * n_agents, n_head, hidden_dim]
        # Pool: Sum
        # embedding_enemies = embedding_enemies.sum(dim=1, keepdim=False)  # [bs * n_agents, n_heads, rnn_hidden_dim]

        # (4) Ally features
        hyper_ally_out = self.hyper_ally(ally_feats_t)
        if self.args.map_type == "MMM":
            # [bs * n_agents * n_allies, ally_fea_dim, rnn_hidden_dim * head]
            fc1_w_ally = hyper_ally_out[:, :-(self.rnn_hidden_dim + 1) * self.n_heads].reshape(
                -1, self.ally_feats_dim, self.rnn_hidden_dim * self.n_heads
            )
        else:
            # [bs * n_agents * n_allies, ally_fea_dim, rnn_hidden_dim * head]
            fc1_w_ally = hyper_ally_out.view(-1, self.ally_feats_dim, self.rnn_hidden_dim * self.n_heads)
        # [bs * n_agents * n_allies, 1, ally_fea_dim] * [bs * n_agents * n_allies, ally_fea_dim, n_heads* rnn_hidden_dim] = [bs * n_agents * n_allies, 1, n_heads*rnn_hidden_dim]
        embedding_allies = th.matmul(ally_feats_t.unsqueeze(1), fc1_w_ally).view(
            bs * self.n_agents, self.n_allies, self.n_heads, self.rnn_hidden_dim
        )  # [bs * n_agents, n_allies, head, rnn_hidden_dim]

        # Pool: Weight_Sum (Attention)
        embedding_allies = (embedding_allies.permute(0, 2, 1, 3)
                            .reshape(bs * self.n_agents * self.n_heads, self.n_allies, -1))
        embedding_allies = self.attention_ally(
            embedding_own.unsqueeze(dim=1).repeat(self.n_heads, 1, 1),  # K: [bs * n_agents * n_head, 1, hidden_dim]
            embedding_allies,  # Q: [bs * n_agents * n_head, n_enemies, hidden_dim]
            embedding_allies,  # V: [bs * n_agents * n_head, n_enemies, hidden_dim]
            # mask_enemies.repeat(self.n_heads, 1, 1)  # MASK: [bs * n_agents * n_head, 1, n_allies]
        ).view(bs * self.n_agents, self.n_heads, -1)  # [bs * n_agents, n_head, hidden_dim]
        # Pool: Sum
        # embedding_allies = embedding_allies.sum(dim=1, keepdim=False)  # [bs * n_agents, head, rnn_hidden_dim]

        # Final embedding
        embedding = embedding_own + self.unify_input_heads(
            embedding_enemies + embedding_allies
        )  # [bs * n_agents, head, rnn_hidden_dim]


        return embedding, hyper_enemy_out, hyper_ally_out

def kaiming_uniform_(tensor_w, tensor_b, mode='fan_in', gain=12 ** (-0.5)):
    fan = nn.init._calculate_correct_fan(tensor_w.data, mode)
    std = gain / math.sqrt(fan)
    bound_w = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    bound_b = 1 / math.sqrt(fan)
    with th.no_grad():
        tensor_w.data.uniform_(-bound_w, bound_w)
        if tensor_b is not None:
            tensor_b.data.uniform_(-bound_b, bound_b)

def get_activation_func(name, hidden_dim):
    """
    'relu'
    'tanh'
    'leaky_relu'
    'elu'
    'prelu'
    :param name:
    :return:
    """
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "tanh":
        return nn.Tanh()
    elif name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    elif name == "elu":
        return nn.ELU(alpha=1., inplace=True)
    elif name == 'prelu':
        return nn.PReLU(num_parameters=hidden_dim, init=0.25)

class Hypernet(nn.Module):
    def __init__(self, input_dim, hidden_dim, main_input_dim, main_output_dim, activation_func, n_heads):
        super(Hypernet, self).__init__()

        self.n_heads = n_heads
        # the output dim of the hypernet
        output_dim = main_input_dim * main_output_dim
        # the output of the hypernet will be reshaped to [main_input_dim, main_output_dim]
        self.main_input_dim = main_input_dim
        self.main_output_dim = main_output_dim

        self.multihead_nn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            get_activation_func(activation_func, hidden_dim),
            nn.Linear(hidden_dim, output_dim * self.n_heads),
        )

    def forward(self, x):
        # [...,  main_output_dim + main_output_dim + ... + main_output_dim]
        # [bs, main_input_dim, n_heads * main_output_dim]
        return self.multihead_nn(x).view([-1, self.main_input_dim, self.main_output_dim * self.n_heads])

class Merger(nn.Module):
    def __init__(self, head, fea_dim):
        super(Merger, self).__init__()
        self.head = head
        if head > 1:
            self.weight = Parameter(th.Tensor(1, head, fea_dim).fill_(1.))
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        :param x: [bs, n_head, fea_dim]
        :return: [bs, fea_dim]
        """
        if self.head > 1:
            return th.sum(self.softmax(self.weight) * x, dim=1, keepdim=False)
        else:
            return th.squeeze(x, dim=1)
