import argparse
from motmodelvshuman import Stimuli, HumanResponses, AVAILABLE_EXPERIMENTS
from motmodelvshuman import MotModel, evaluate_models, AVAILABLE_MODELS

MODEL_NAMES = AVAILABLE_MODELS
EXPERIMENT_NAMES = AVAILABLE_EXPERIMENTS

def main(args):
    out_path = './results'
    experiment_name = 'Experiment1'

    models = [MotModel(model_name) for model_name in args.models]
    stimuli = Stimuli(args.experiment_name)
    human_responses = HumanResponses(args.experiment_name)

    evaluate_models(models, stimuli, human_responses, out_path=out_path)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=MODEL_NAMES, help='models to evaluate', choices=MODEL_NAMES)
    parser.add_argument('--experiment_name', default='Experiment1', help='experiment name', choices=AVAILABLE_EXPERIMENTS)
    args = parser.parse_args()
    
    main(args)