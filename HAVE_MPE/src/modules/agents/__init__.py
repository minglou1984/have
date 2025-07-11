REGISTRY = {}

from .rnn_agent import RNNAgent
from .tree_rnn_agent import TreeRNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["tree_rnn"] = TreeRNNAgent
