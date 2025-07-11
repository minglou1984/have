from .q_learner import QLearner
from .qtran_learner import QLearner as QTranLearner
REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["qtran_learner"] = QTranLearner
