import torch as th
import torch.nn as nn
from torch.nn.parameter import Parameter
from modules.layer.attention import MultiHeadAttention
from modules.layer.attention import MHA

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
    def __init__(self, args, input_dim, hidden_dim, main_input_dim, main_output_dim, activation_func, n_heads):
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
        return self.multihead_nn(x).view([-1, self.main_input_dim, self.n_heads * self.main_output_dim])

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

class TreeEncoder(nn.Module):
    def __init__(self, input_shape, output_shape, args):
        super(TreeEncoder, self).__init__()
        self.args = args

        self.n_agents = args.n_agents
        self.n_allies = args.n_allies
        self.n_enemies = args.n_enemies
        self.n_actions = args.n_actions

        self.n_heads = args.hpn_head_num
        self.entity_hidden_dim = output_shape
        self.hpn_attention_head = args.hpn_attention_head

        # [4, (4, 2), (5, 2)]
        self.own_feats_dim, self.ally_feats_dim, self.enemy_feats_dim = input_shape
        self.enemy_feats_dim = self.enemy_feats_dim[-1]  # [n_enemies, feat_dim]
        self.ally_feats_dim = self.ally_feats_dim[-1]  # [n_allies, feat_dim]

        # 1.Unique Features (do not need hyper net)
        self.fc1_own = nn.Linear(self.own_feats_dim, self.entity_hidden_dim, bias=True)  # only one bias is OK

        # 2.Hypernet-based API input layer
        # Multiple entities (use hyper net to process these features to ensure permutation invariant)
        self.hyper_input_w_enemy = Hypernet(args=args,
            input_dim=self.enemy_feats_dim, hidden_dim=args.hpn_hyper_dim,
            main_input_dim=self.enemy_feats_dim, main_output_dim=self.entity_hidden_dim,
            activation_func=args.hpn_hyper_activation, n_heads=self.n_heads
        )  # output shape: (enemy_feats_dim * self.entity_hidden_dim * n_heads)
        self.hyper_input_w_ally = Hypernet(args=args,
            input_dim=self.ally_feats_dim, hidden_dim=args.hpn_hyper_dim,
            main_input_dim=self.ally_feats_dim, main_output_dim=self.entity_hidden_dim,
            activation_func=args.hpn_hyper_activation, n_heads=self.n_heads
        )  # output shape: (ally_feats_dim * hidden_dim * n_heads)

        # 3.Attention layer
        self.attention_ally = MultiHeadAttention(self.entity_hidden_dim, heads=self.hpn_attention_head)
        self.attention_enemy = MultiHeadAttention(self.entity_hidden_dim, heads=self.hpn_attention_head)
        # self.attention_ally = MHA(self.entity_hidden_dim, heads=self.hpn_attention_head)
        # self.attention_enemy = MHA(self.entity_hidden_dim, heads=self.hpn_attention_head)

        # self.unify_input_heads = nn.Linear(self.entity_hidden_dim * self.n_heads, self.entity_hidden_dim)
        self.unify_input_heads = Merger(self.n_heads, self.entity_hidden_dim)

    def forward(self, inputs):
        # bs: batch_size for actor, batch_size * max_t for critic
        # own_feats_t: [bs * n_agents, own_fea_dim],
        # ally_feats_t: [bs * n_agents * n_allies, ally_fea_dim]
        # enemy_feats_t: [bs * n_agents * n_enemies, enemy_fea_dim],
        bs, own_feats_t, ally_feats_t, enemy_feats_t = inputs

        # (1) Own features, [bs * n_agents, own_fea_dim] => [bs * n_agents, hidden_dim]
        embedding_own = self.fc1_own(own_feats_t)

        # (2) Ally features: [bs * n_agents * n_allies, ally_fea_dim] => [bs * n_agents, n_head, hidden_dim]
        # Generate Weight,
        input_w_ally = self.hyper_input_w_ally(ally_feats_t) # [bs * n_agents * n_allies, ally_fea_dim, head, hidden_dim]
        # Linear transform,
        embedding_allies = th.matmul(
            ally_feats_t.unsqueeze(1),
            input_w_ally
        ).view(bs * self.n_agents, self.n_allies, self.n_heads, -1) # [bs * n_agents, n_allies, head, hidden_dim]

        # Pool, sum, weight_sum, mean
        embedding_allies = embedding_allies.permute(0, 2, 1, 3).reshape(bs * self.n_agents * self.n_heads, self.n_allies, -1)
        # .contiguous().view(bs * self.n_agents * self.n_heads, self.n_allies, -1) # [bs * n_agents * n_head, n_allies, hidden_dim]
        mask_allies = (th.abs(ally_feats_t).sum(dim=-1, keepdim=True) !=0 ).reshape(-1, 1, self.n_allies) # [bs * n_agents, 1, n_allies]
        embedding_allies = self.attention_ally(
            embedding_own.unsqueeze(dim=1).repeat(self.n_heads, 1, 1),  # K: [bs * n_agents * n_head, 1, hidden_dim]
            embedding_allies,                                           # Q: [bs * n_agents * n_head, n_allies, hidden_dim]
            embedding_allies,                                           # V: [bs * n_agents * n_head, n_allies, hidden_dim]
            mask_allies.repeat(self.n_heads, 1, 1)                      # MASK: [bs * n_agents * n_head, 1, n_allies]
        ).view(bs * self.n_agents, self.n_heads, -1) # [bs * n_agents, n_head, hidden_dim]

        # (3) Enemy features: [bs * n_agents * n_enemies, enemy_fea_dim] -> [bs * n_agents, n_head, hidden_dim]
        # Generate Weight,
        input_w_enemy = self.hyper_input_w_enemy(enemy_feats_t) # [bs * n_agents * n_enemies, enemy_fea_dim, head, hidden_dim]
        # Linear transform,
        embedding_enemies = th.matmul(
            enemy_feats_t.unsqueeze(1),
            input_w_enemy
        ).view(bs * self.n_agents, self.n_enemies, self.n_heads, -1) # [bs * n_agents, n_enemies, head, hidden_dim]
        # Pool: sum, weight_sum, mean
        embedding_enemies = embedding_enemies.permute(0, 2, 1, 3).reshape(bs * self.n_agents * self.n_heads, self.n_enemies, -1)
            # .contiguous().view(bs * self.n_agents * self.n_heads, self.n_enemies, -1)  # [bs * n_agents * n_head, n_enemies, hidden_dim]
        mask_enemies = (th.abs(enemy_feats_t).sum(dim=-1, keepdim=True) != 0).reshape(-1, 1, self.n_enemies)  # [bs * n_agents, 1, n_enemies]
        embedding_enemies = self.attention_enemy(
            embedding_own.unsqueeze(dim=1).repeat(self.n_heads, 1, 1),  # K: [bs * n_agents * n_head, 1, hidden_dim]
            embedding_enemies,                                          # Q: [bs * n_agents * n_head, n_allies, hidden_dim]
            embedding_enemies,                                          # V: [bs * n_agents * n_head, n_allies, hidden_dim]
            mask_enemies.repeat(self.n_heads, 1, 1)                    # MASK: [bs * n_agents * n_head, 1, n_allies]
        ).view(bs * self.n_agents, self.n_heads, -1) # [bs * n_agents, n_head, hidden_dim]

        # (4) Fusion
        embedding = embedding_own + self.unify_input_heads(
            embedding_enemies + embedding_allies
        ) # [bs * n_agents, hidden_dim]
        return embedding
