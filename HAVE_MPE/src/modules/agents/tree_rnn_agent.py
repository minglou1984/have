import torch.nn as nn
import torch.nn.functional as F
from modules.layer.permutation import TreeEncoder

class TreeRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(TreeRNNAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.hidden_dim = args.hidden_dim
        self.TreeEncoder = TreeEncoder(input_shape, args.hidden_dim, args)
        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.fc2_basic_actions_stream = nn.Linear(self.hidden_dim, args.n_actions)  # 5(stop, up, down, right, left)
        # self.initialize_weights()

    def initialize_weights(self):
        # use Xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
            elif isinstance(m, nn.GRUCell):
                nn.init.orthogonal_(m.weight_ih)
                nn.init.orthogonal_(m.weight_hh)
                # nn.init.zeros_(m.bias_ih)
                # nn.init.zeros_(m.bias_hh)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc2_basic_actions_stream.weight.new(1, self.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.TreeEncoder(inputs))
        h = hidden_state.reshape(-1, self.hidden_dim)
        h = self.rnn(x, h) # [bs * n_agents, hidden_dim]
        # Q-values of basic actions
        q = self.fc2_basic_actions_stream(h) # [bs*n_agents, n_actions]
        return q, h
