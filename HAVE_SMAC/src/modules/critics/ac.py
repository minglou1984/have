import torch as th
import torch.nn as nn
import torch.nn.functional as F
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm

class ACCritic(nn.Module):
    def __init__(self, scheme, groups, args):
        super(ACCritic, self).__init__()

        self.args = args
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)

        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)

        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

    def init_hidden(self, batch_size):
        # make hidden states on same device as model
        self.hidden_states = self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def forward(self, batch, t=None):
        inputs, bs = self._build_inputs(batch, t=t)

        x = F.relu(self.fc1(inputs), inplace=True)
        h_in = self.hidden_states.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)

        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(h))
        else:
            q = self.fc2(h)
        self.hidden_states = h
        return q.view(bs, self.n_agents, -1)

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        inputs = []
        # observations
        inputs.append(batch["obs"][:, t])
        # last actions
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        # agent id
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=-1)
        return inputs, bs

    def _get_input_shape(self, scheme):
        # observations
        input_shape = scheme["obs"]["vshape"]
        # last actions
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        # agent id
        input_shape += self.n_agents
        return input_shape
