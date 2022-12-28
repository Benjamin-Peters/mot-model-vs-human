from motmodelvshuman import MotModel
from motmodelvshuman import Stimuli, HumanResponses
from motmodelvshuman import evaluate_models

out_path = './results'
experiment_name = 'experiment1'

models = [MotModel('sort'), MotModel('deepsort')]
stimuli = Stimuli(experiment_name)
human_responses = HumanResponses(experiment_name)

evaluate_models(models, stimuli, human_responses, out_path=out_path)