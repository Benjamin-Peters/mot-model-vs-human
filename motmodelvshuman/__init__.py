from .data import Stimuli, HumanResponses, ModelOutput, AVAILABLE_EXPERIMENTS
from .evaluation import evaluate_models


AVAILABLE_MODELS = ['deepsort', 'sort', 'ocsort', 'bytetrack']
ADDITIONAL_MODEL_IDS = ['nomotiongating', 'nomotiongating_gtreid'] #TODO