```python

from motmodelvshuman import MotModel, evaluate_model
from motmodelvshuman import Stimuli
from motmodelvshuman import HumanResponses

models = [MotModel('sort'), MotModel('deepsort')]
stimuli = Stimuli('experiment1')

for model in models:
	out_path = './results'
	# results will be stored at out_path / stimuli.name / model.name
	model_output = evaluate_model(model, stimuli, out_path=out_path)
	model_responses = response_from_output(model_output)
	
human_responses = HumanData('experiment1')

# plots

# statistics


```