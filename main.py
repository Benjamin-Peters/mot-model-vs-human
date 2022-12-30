import argparse
from motmodelvshuman import Stimuli, HumanResponses, ModelOutput
from motmodelvshuman import evaluate_models
from motmodelvshuman import AVAILABLE_MODELS, ADDITIONAL_MODEL_IDS

def main(models, experiment_name, additional_model_ids):
    out_path = './results'

    model_outputs = [ModelOutput(experiment=experiment_name, model_name=model_name, additional_model_id=additional_model_id) for model_name in models for additional_model_id in additional_model_ids]
    stimuli = Stimuli(experiment_name)
    human_responses = HumanResponses(experiment_name)

    evaluate_models(model_outputs, stimuli, human_responses, out_path=out_path)
    
    
if __name__ == '__main__':   
    
    # Figure 2:
    main(AVAILABLE_MODELS, 'experiment1')
    # Figure 3:
    main(AVAILABLE_MODELS, 'experiment2')
    # Figure 4:
    main(['deepsort'], 'experiment1', ADDITIONAL_MODEL_IDS)
    # Supplementary Figure 1:
    # TODO