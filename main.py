from motmodelvshuman import Stimuli, HumanResponses, ModelOutput, evaluate_models
from motmodelvshuman import MAIN_MODELS, GATING_MODELS, NOISY_REID_MODELS


MODEL_NAMES = {
    'experiment1': MAIN_MODELS,
    'experiment2': MAIN_MODELS,
    'experiment1_gating': GATING_MODELS,
    'experiment1_noisy_reid': NOISY_REID_MODELS
}

STIMULI_HUMAN_RESPONSES = {
    'experiment1': 'experiment1',
    'experiment2': 'experiment2',
    'experiment1_gating': 'experiment1',
    'experiment1_noisy_reid': 'experiment1'
}

def main(experiment):
    model_names = MODEL_NAMES[experiment]
    model_outputs = [ModelOutput(experiment=experiment, model_name=model_name) for model_name in model_names]

    stimuli = Stimuli(experiment=STIMULI_HUMAN_RESPONSES[experiment])
    human_responses = HumanResponses(experiment=STIMULI_HUMAN_RESPONSES[experiment])
    evaluate_models(experiment, model_outputs, stimuli, human_responses, out_path='./results')


if __name__ == '__main__':
    # Figure 2
    print('==== Experiment 1 ====')
    main('experiment1')

    # Figure 3
    print('==== Experiment 2 ====')
    main('experiment2')

    # Figure 4 part 1
    print('==== Experiment 1 (gating) ====')
    main('experiment1_gating')

    # Figure 4 part 2
    print('==== Experiment 1 (noisy reid) ====')
    main('experiment1_noisy_reid')
    
