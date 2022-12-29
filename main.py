import argparse
from motmodelvshuman import Stimuli, HumanResponses, ModelOutput, AVAILABLE_EXPERIMENTS
from motmodelvshuman import MotModel, evaluate_models, AVAILABLE_MODELS

MODEL_NAMES = AVAILABLE_MODELS
EXPERIMENT_NAMES = AVAILABLE_EXPERIMENTS

def main(args):
    out_path = './results'

    model_outputs = [ModelOutput(experiment=args.experiment_name, model_name=model_name) for model_name in args.models]
    stimuli = Stimuli(args.experiment_name)
    human_responses = HumanResponses(args.experiment_name)

    evaluate_models(model_outputs, stimuli, human_responses, out_path=out_path)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=MODEL_NAMES, help='models to evaluate', choices=MODEL_NAMES)
    parser.add_argument('--experiment_name', default='experiment1', help='experiment name', choices=AVAILABLE_EXPERIMENTS)
    args = parser.parse_args()
    
    main(args)