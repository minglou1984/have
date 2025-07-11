from functools import partial
from .multiagentenv import MultiAgentEnv
from .mpe.mpe_wrapper import MPEWrapper

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["mpe"] = partial(env_fn, env=MPEWrapper)
