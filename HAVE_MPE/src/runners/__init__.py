REGISTRY = {}

from .episode_runner import EpisodeRunner
from .gymma_episode_runner import GymmaEpisodeRunner
from .parallel_runner import ParallelRunner
from .gymma_parallel_runner import GymmaParallelRunner

REGISTRY["episode"] = EpisodeRunner
REGISTRY["gymma_episode"] = GymmaEpisodeRunner
REGISTRY["parallel"] = ParallelRunner
REGISTRY["gymma_parallel"] = GymmaParallelRunner

