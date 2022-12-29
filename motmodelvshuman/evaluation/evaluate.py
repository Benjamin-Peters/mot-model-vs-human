from typing import Union

import pandas as pd
import numpy as np

# models = [MotModel('sort'), MotModel('deepsort')]
# stimuli = Stimuli('experiment1')

from ..data import Stimuli, HumanResponses
from ..models import MotModel
from ..models import model_inference, MotModel
from .plots import plot_accuracy

from pathlib import Path
LOCAL_MODEL_OUTPUTS = Path('./data/model_outputs')

# this is to screen participants who finished the experiment
# (the number is slightly smaller than the actual number of trials because
# sometimes a couple of trials would not be recorded properly)
N_TRIALS_SCREEN_SUBJECTS = {
    'experiment1': 107,
    'experiment2': 93,
}
MAX_N_TARGETS = 4

class ModelResponses(pd.DataFrame):
    pass

def response_from_output(model_output) -> ModelResponses:
    """
        Convert model output to human-like responses
    """
    # needto use code from here?
    # Python/evaluation/model_output/tracker_output_to_fo_responses.py
    pass

def get_factors(human_responses:HumanResponses) -> dict:
    df = human_responses.df
    factors = {}
    print(df)
    factors['n_valued_objects'] = np.sort(df.n_valued_objects.unique())
    factors['occlusion_levels'] = np.sort(df.occlusion_levels.unique())
    factors['category_similarity'] = np.sort(df.category_similarity.unique())
    if 'n_distractor_objects' in df.columns: # this is only in experiment2
        factors['n_distractor_objects'] = np.sort(df.n_distractor_objects.unique())
    return factors




def compute_human_accuracy(human_responses:HumanResponses, factors:dict) -> tuple:
    """
        Compute human accuracy for each factor level and return the mean accuracy
        as well as the number of subjects that pass the screening criteria
    """
    df = human_responses.df
    experiment = human_responses.experiment
    n_trials_screen_subjects = N_TRIALS_SCREEN_SUBJECTS[experiment]

    # selecting subjects
    selected_subjects = []
    for user_session_id in df.index.get_level_values(0).unique():
        print(user_session_id)
        n_trials_done = len(df.xs(user_session_id, level=0))
        if n_trials_done >= n_trials_screen_subjects:
            selected_subjects.append(user_session_id)
        print(f"{user_session_id} n_trials done: {n_trials_done}")

    cols = [f'correct_{i}' for i in range(MAX_N_TARGETS)]
    df['correct'] = df[cols].mean(axis=1, skipna=True)

    means_dims = tuple([len(levels) for levels in factors.values()])
    means = np.zeros((len(selected_subjects), *means_dims))

    for index in np.ndindex((len(selected_subjects), *means_dims)):
        row_sel = df.index.get_level_values(0) == selected_subjects[index[0]]
        for j, (factor, levels) in enumerate(factors.items()):
            row_sel = row_sel & (df[factor] == levels[index[j+1]])
        means[index] = df[row_sel]['correct'].mean()

    return means, len(selected_subjects)


def compute_model_accuracy(model_responses: ModelResponses, factors:dict) -> tuple:
    """
        Compute model accuracy for each factor level and return the mean accuracy
        as well as the number of model runs
    """
    df = model_responses.df

    cols = [f'correct_{i}' for i in range(MAX_N_TARGETS)]
    df['correct'] = df[cols].mean(axis=1, skipna=True)

    n = len(df.run.unique())
    # print([len(levels) for (factor, levels) in factors])
    print(factors.keys())
    means_dims = tuple([len(levels) for levels in factors.values()])
    means = np.zeros((n, *means_dims))

    for index in np.ndindex((n, *means_dims)):
        row_sel = df.run == index[0]
        for j, (factor, levels) in enumerate(factors.items()):
            row_sel = row_sel & (df[factor] == levels[index[j+1]])
        means[index] = df[row_sel]['correct'].mean()

    return means, n


def get_model_responses(model:MotModel, stimuli:Stimuli, factors:dict, out_path:str='./results'):
    """
        1) run inference to produce model output
        2) convert to human-like responses (from NeuroArcade/evaluation/model_output)
        3) compute metrics for models and humans
        4) save metrics
    """
    
    # 1) run model inference (results will be stored at out_path / stimuli.name / model.name)
    # model_output = model_inference(model, stimuli, out_path=full_out_path)
    model_output_path = LOCAL_MODEL_OUTPUTS / stimuli.experiment / f"{model.name}_{stimuli.experiment}.pkl"
    model_output = pd.read_pickle(model_output_path)

    # 2) convert model output to human-like responses
    model_output_path_csv = Path(out_path) / stimuli.experiment / f"{model.name}_{stimuli.experiment}.csv"
    model_responses = response_from_output(model_output, out_path = model_output_path_csv)
    
    
    #metrics = {'model': model.name,  'accuracy': model_accuracy}
        
    # 4) save metrics
    # TODO save metrics to full_out_path
    
    return model_accuracy, n_runs

def evaluate_models(models:list[MotModel], stimuli:Stimuli, human_responses: HumanResponses, out_path:str='./results'):
    """
    
        1) compute human accuracy
        2) for each model: compute accuracy via evaluate_model
        3) create and save plots
        4) compute statistics
    
    """

    assert stimuli.experiment == human_responses.experiment, "Stimuli and human responses must be from the same experiment"
    
    # 1) compute human accuracy
    factors = get_factors(human_responses)
    accuracy = {}
    n = {}
    accuracy["human"], n["human"] = compute_human_accuracy(human_responses, factors)
    
    # 2) for each model: get responses from mmtracking data and compute accuracy
    # for model in models:
    #     model_responses = get_model_responses(model, stimuli, factors, out_path=out_path)
    #     accuracy[model.name], n[model.name] = compute_model_accuracy(model_responses, factors)
        #metrics = evaluate_model(model, stimuli, human_responses, out_path=out_path)
        #metrics_list.append(metrics)
    
    # 3) create and save plots
    plot_accuracy(stimuli.experiment, accuracy, n, factors, out_path=out_path)
    
    # 4) compute statistics
    # TODO
        
    pass