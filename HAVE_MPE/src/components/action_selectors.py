import torch as th
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule

def categorical_entropy(probs):
    assert probs.size(-1) > 1
    return Categorical(probs=probs).entropy()

REGISTRY = {}

class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions

REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = getattr(self.args, "evaluation_epsilon", 0.0)

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions

    def update_epsilon(self, t_env):
        self.epsilon = self.schedule.eval(t_env)
        return self.epsilon

REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector


class SoftPoliciesSelector():

    def __init__(self, args):
        self.args = args

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        m = Categorical(agent_inputs)
        picked_actions = m.sample().long()
        return picked_actions

REGISTRY["soft_policies"] = SoftPoliciesSelector


class GumbelSoftmax:

    def __init__(self, logits, temperature=1):
        super(GumbelSoftmax, self).__init__()
        self.logits = logits
        self.eps = 1e-20
        self.temperature = temperature

    # gumbel noise
    def sample_gumbel(self):
        tens_type = type(self.logits.data)
        U = Variable(tens_type(self.logits.shape).uniform_(), requires_grad=False)
        return -th.log(-th.log(U + self.eps) + self.eps)

    # softmax(logits+gumbel noise)
    def gumbel_softmax_sample(self):
        y = self.logits + self.sample_gumbel().to(self.logits.device)
        return F.softmax( y / self.temperature, dim=-1)

    def onehot_from_logits(self, logits=None):
        """
        Given batch of logits, return one-hot sample using epsilon greedy strategy
        (based on given epsilon)
        """
        # get best (according to current policy) actions in one-hot form
        logits = self.logits if logits is None else logits
        argmax_acs = (logits == logits.max(-1, keepdim=True)[0]).float()
        return argmax_acs

    # Gumbel-Softmax Trick + Straight-Through Estimator
    # modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
    def gumbel_softmax(self, hard=False):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
          logits: [batch_size, n_class] unnormalized log-probs
          temperature: non-negative scalar
          hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
          [batch_size, n_class] sample from the Gumbel-Softmax distribution.
          If hard=True, then the returned sample will be one-hot, otherwise it will
          be a probabilitiy distribution that sums to 1 across classes
        """
        y = self.gumbel_softmax_sample()
        if hard:
            y_hard = self.onehot_from_logits(y)
            y = (y_hard - y).detach() + y
        return y


class GumbelSoftmaxMultinomialActionSelector():

    def __init__(self, args):
        self.args = args

    def gumbel_softmax_pi(self, agent_inputs):
        gumbel_action = GumbelSoftmax(logits=agent_inputs)
        pi = gumbel_action.gumbel_softmax(hard=True)
        return pi

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        gumbel_action = GumbelSoftmax(logits=agent_inputs)
        if test_mode:
            chosen_actions = gumbel_action.onehot_from_logits()
        else:
            chosen_actions = gumbel_action.gumbel_softmax(hard=True).argmax(dim=-1)

        return chosen_actions

REGISTRY["gumbel_softmax_multinominal"] = GumbelSoftmaxMultinomialActionSelector


class GaussianActionSelector():

    def __init__(self, args):
        self.args = args
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, mu, sigma, test_mode=False):
        # Expects the following input dimensions:
        # mu: [b x a x u]
        # sigma: [b x a x u x u]
        assert mu.dim() == 3, "incorrect input dim: mu"
        assert sigma.dim() == 3, "incorrect input dim: sigma"
        sigma = sigma.view(-1, self.args.n_agents, self.args.n_actions, self.args.n_actions)

        if test_mode and self.test_greedy:
            picked_actions = mu
        else:
            dst = th.distributions.MultivariateNormal(mu.view(-1,mu.shape[-1]),
                                                      sigma.view(-1, mu.shape[-1], mu.shape[-1]))
            picked_actions = dst.sample().view(*mu.shape)
        return picked_actions

REGISTRY["gaussian"] = GaussianActionSelector