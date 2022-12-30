import argparse
from motmodelvshuman import Stimuli, HumanResponses, ModelOutput
from motmodelvshuman import evaluate_models
from motmodelvshuman import AVAILABLE_EXPERIMENTS, AVAILABLE_MODELS

def main(args):
    out_path = './results'

    model_outputs = [ModelOutput(experiment=args.experiment_name, model_name=model_name) for model_name in args.models]
    stimuli = Stimuli(args.experiment_name)
    human_responses = HumanResponses(args.experiment_name)

    evaluate_models(model_outputs, stimuli, human_responses, out_path=out_path)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=AVAILABLE_MODELS, help='models to evaluate', choices=AVAILABLE_MODELS)
    parser.add_argument('--experiment_name', default='experiment1', help='experiment name', choices=AVAILABLE_EXPERIMENTS)
    parser.add_argument('--additional_models', action='store_true', help='whether to evaluate the additional models on experiment1')
    args = parser.parse_args()
    assert not args.additional_models or args.experiment_name == 'experiment1', "Additional models are only available for experiment1"
    main(args)