from typing import Union

import pandas as pd

# models = [MotModel('sort'), MotModel('deepsort')]
# stimuli = Stimuli('experiment1')

from ..data import Stimuli, HumanResponses
from ..models import MotModel
from ..models import model_inference, MotModel

class ModelResponses(pd.DataFrame):
    pass

def response_from_output(model_output) -> ModelResponses:
    """
        Convert model output to human-like responses
    """
    pass

def compute_accuracy(responses: HumanResponses | ModelResponses) -> pd.DataFrame:
    """
        Compute accuracy for each model and human
    """
    pass

def evaluate_model(model:MotModel, stimuli:Stimuli, human_responses: HumanResponses, out_path:str='./results'):
    """

        1) run inference to produce model output
        2) convert to human-like responses (from NeuroArcade/evaluation/model_output)
        3) compute metrics for models and humans
        4) save metrics
    
    """
    
    assert stimuli.experiment == human_responses.experiment, "Stimuli and human responses must be from the same experiment"
    
    full_out_path = out_path / stimuli.experiment / model.name
    
    # 1) run model inference (results will be stored at out_path / stimuli.name / model.name)
    model_output = model_inference(model, stimuli, out_path=full_out_path)
    
    # 2) convert model output to human-like responses
    model_responses = response_from_output(model_output, out_path = full_out_path)
    
    # 3) compute metrics for models and humans
    model_accuracy = compute_accuracy(model_responses)
    
    metrics = {'model': model.name,  'accuracy': model_accuracy}
        
    # 4) save metrics
    # TODO save metrics to full_out_path
    
    return metrics

def evaluate_models(models:list[MotModel], stimuli:Stimuli, human_responses: HumanResponses, out_path:str='./results'):
    """
    
        1) compute human accuracy
        2) for each model: compute accuracy via evaluate_model
        3) create and save plots
        4) compute statistics
    
    """
    
    # 1) compute human accuracy
    human_accuracy = compute_accuracy(human_responses)
    
    # 2) for each model: compute accuracy via evaluate_model
    metrics_list = []
    for model in models:
        metrics = evaluate_model(model, stimuli, human_responses, out_path=out_path)
        metrics_list.append(metrics)
    
    # 3) create and save plots
    # TODO
    
    # 4) compute statistics
    # TODO
        
    pass