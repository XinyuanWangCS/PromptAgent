from . import *
from .beam_search import BeamSearch
from .mcts import MCTS

SEARCH_ALGOS = {'beam_search': BeamSearch,
                'mcts' : MCTS}

def get_search_algo(algo_name):
    assert algo_name in SEARCH_ALGOS.keys(), f"Search algo {algo_name} is not supported."
    return SEARCH_ALGOS[algo_name]