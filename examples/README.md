```python

from motmodelvshuman import MotModel
from motmodelvshuman import Stimuli, HumanResponses
from motmodelvshuman import evaluate_models

out_path = './results'
experiment_name = 'Experiment1'

models = [MotModel('sort'), MotModel('deepsort')]
stimuli = Stimuli(experiment_name)
human_responses = HumanData(experiment_name)

evaluate_models(models, stimuli, human_responses, out_path=out_path)


```