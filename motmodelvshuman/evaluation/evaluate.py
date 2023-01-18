from typing import Union, List
import json
import pickle

import pandas as pd
import numpy as np

from ..data import Stimuli, HumanResponses, ModelOutput
from .response_models import simple_response as response_model
from .plots import plot_accuracy
from .stats import calculate_anovas

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

MODEL_OUTPUT_COLUMNS = ['bpid', 
                'frame_idx', 
                'obj_id', 
                'bbox_left', 'bbox_top',
                'bbox_w','bbox_h',
                'pred_conf']

def format_model_output(model_output: ModelOutput,
                      annotation_file_path: Path,
                      ) -> pd.DataFrame :
    
    # load annotations from stimuli
    annotations_json = json.load(open(annotation_file_path, 'r'))
    
    videos = annotations_json['videos']
    video_ids = [vid['id'] for vid in videos]
    video_names = {video['id']:video['name'] for video in videos}
    
    assert len(np.unique(list(video_names.values()))) == len(list(video_names.values())), "video names need to be unique (the video name should be the unique blueprint id)"
    for vid_name in video_names.values():
        assert vid_name.startswith('BPID'), f'video name {vid_name} does not start with BPID (seems like not part of an experiment)'
    
    images_by_video_id = {}
    for vid in video_ids:
        images_by_video_id[vid] = [x for x in annotations_json['images'] if x['video_id'] == vid]
    video_lengths = {vid: len(images_by_video_id[vid]) for vid in video_ids}
    
    video_name_per_image = [video_names[image['video_id']] for image in annotations_json['images']]
    mot_frame_id_per_image = [image['mot_frame_id'] for image in annotations_json['images']]
    
    # images_by_video_id[vid] contains 'id', 'video_id', 'file_name', 'width', 'height', 'frame_id', 'mot_frame_id'
    
    # make sure the video starts with the first frame (mot_frame_id = 0) 
    # - this is for flying objects only and to make sure that the full video
    # has been evaluated (not just a subset of it - when using "half-")    
    for vid in video_ids:
        # if images_by_video_id[vid][0]['mot_frame_id'] != 0:
        if images_by_video_id[vid][0]['mot_frame_id'] != 1:
            err_str = f"video {vid} does not start with frame 1 - you may need to regenerate the dataset with 1-based frame indexing (warning in evaluation.model_output.tracker_output_to_fo_responses.mmtracking_to_tracker_output)"
            raise ValueError(err_str)
        
    # 2) load and parse results
    assert len(model_output.data['track_bboxes']) == sum(video_lengths.values()), f"number of frames in results ({len(model_output.data['track_bboxes'])}) does not match number of frames in annotations ({sum(video_lengths.values())}) for files {annotation_file_path}"
        
    rows = []
    model_output_formatted = pd.DataFrame(columns=MODEL_OUTPUT_COLUMNS +['score'])
    for frame_idx, frame, video_name in zip(mot_frame_id_per_image, model_output.data['track_bboxes'], video_name_per_image):
        for detection in frame[0]:
            track_id, bb_left, bb_top, bb_right, bb_bottom, score = detection # for some reason, the mmtracking does not output bbox width and height
            bb_width = bb_right - bb_left
            bb_height = bb_bottom - bb_top
            # track = [x1, y1, x2, y2, score, track_id]
            row = dict(bpid=video_name,
                       frame_idx=frame_idx,
                       obj_id=track_id,
                       bbox_left=bb_left,
                       bbox_top=bb_top,
                       bbox_w=bb_width,
                       bbox_h=bb_height,
                       pred_conf=score)
            rows.append(row)
    model_output_formatted = model_output_formatted.append(rows, ignore_index=True)
    model_output_formatted.model_name = model_output.model_name
    return model_output_formatted
            
def response_from_output(model_output_formatted: pd.DataFrame, stimuli: Stimuli, n_runs: int = 10, out_path:Path=None) -> ModelResponses:
    """
        Convert model output to human-like responses
    """
    
    with open(stimuli.experimental_session_file, 'r') as fin:
        blueprints = json.load(fin)
    
    responses = response_model(model_output_formatted, 
                    blueprints,
                    n_runs=n_runs, 
                    model_name=model_output_formatted.model_name,
                    experimental_session_id=stimuli.experimental_session_id,
                    first_frame = 20,
                    # last_frame = 179,
                    verbose=False)    
    
    responses['experimental_session_id'] = stimuli.experimental_session_id

    if out_path is not None:
        responses.to_csv(out_path, index=False)
    
    return responses    

def get_factors(human_responses:HumanResponses) -> dict:
    df = human_responses.data
    factors = {}
    factors['n_valued_objects'] = np.sort(df.n_valued_objects.unique())
    factors['occlusion_levels'] = np.sort(df.occlusion_levels.unique())
    factors['category_similarity'] = np.sort(df.category_similarity.unique())
    if human_responses.experiment == 'experiment2': # only experiment 2 has number of distractors as an experimental factor
        factors['n_distractor_objects'] = np.sort(df.n_distractor_objects.unique())
    return factors




def compute_human_accuracy(human_responses:HumanResponses, factors:dict) -> tuple:
    """
        Compute human accuracy for each factor level and return the mean accuracy
        as well as the number of subjects that pass the screening criteria
    """
    df = human_responses.data
    experiment = human_responses.experiment
    n_trials_screen_subjects = N_TRIALS_SCREEN_SUBJECTS[experiment]

    # selecting subjects
    subjects = df.user_session_id.unique()
    selected_subjects = []
    for subject in subjects:
        df_subject = df[df.user_session_id == subject]
        n_trials_done = len(df_subject)
        if n_trials_done >= n_trials_screen_subjects:
            selected_subjects.append(subject)

    cols = [f'correct_{i}' for i in range(MAX_N_TARGETS)]
    df['correct'] = df[cols].mean(axis=1, skipna=True)

    means_dims = tuple([len(levels) for levels in factors.values()])
    means = np.zeros((len(selected_subjects), *means_dims))

    for index in np.ndindex(*means.shape):
        #df_subject = df[df.user_session_id == selected_subjects[index[0]]]
        row_sel = df.user_session_id == selected_subjects[index[0]]
        for j, (factor, levels) in enumerate(factors.items()):
            row_sel = row_sel & (df[factor] == levels[index[j+1]])
        means[index] = df[row_sel]['correct'].mean()

    return means, len(selected_subjects)


def compute_model_accuracy(model_responses: ModelResponses, factors:dict) -> tuple:
    """
        Compute model accuracy for each factor level and return the mean accuracy
        as well as the number of model runs
    """
    df = model_responses

    cols = [f'correct_{i}' for i in range(MAX_N_TARGETS)]
    df['correct'] = df[cols].mean(axis=1, skipna=True)

    n = len(df.run.unique())
    # print([len(levels) for (factor, levels) in factors])
    means_dims = tuple([len(levels) for levels in factors.values()])
    means = np.zeros((n, *means_dims))

    for index in np.ndindex((n, *means_dims)):
        row_sel = df.run == index[0]
        for j, (factor, levels) in enumerate(factors.items()):
            row_sel = row_sel & (df[factor] == levels[index[j+1]])
        means[index] = df[row_sel]['correct'].mean()
    
    return means, n


#def get_model_responses(model_output:ModelOutput, stimuli:Stimuli, factors:dict, out_path:str='./results'):
def get_model_responses(model_output:ModelOutput, stimuli:Stimuli):
    """
        1) convert to human-like responses (from NeuroArcade/evaluation/model_output)
        2) compute metrics for models and humans
        3) save metrics
    """
    
    model_output_path_csv = model_output.data_path.parent / (model_output.data_path.stem + '_responses.csv')
    model_output_formatted = format_model_output(model_output, annotation_file_path = stimuli.annotations_json_path)
    model_responses = response_from_output(model_output_formatted, stimuli, out_path = model_output_path_csv)
    return model_responses


WITHIN_FACTORS = {
    'experiment1': [
        ['n_targets'],
        ['occlusion'],
        ['category_similarity'],
        ['category_similarity', 'occlusion'],
        ['n_targets', 'occlusion'],
    ],
    'experiment2': [
        ['n_targets'],
        ['occlusion'],
        ['category_similarity'],
        ['n_distractors'],
        ['n_targets', 'n_distractors'],
    ] 
}

FIGURE_CONTENTS = {
    'experiment1': {
        "interactions" : [
            ["n_valued_objects", "occlusion_levels"],
            ["category_similarity", "occlusion_levels"]
        ]
    },
    'experiment2': {
        "interactions" : [
            ["n_valued_objects", "occlusion_levels"],
            ["category_similarity", "occlusion_levels"],
            ["n_distractor_objects", "n_valued_objects"],
        ]
    },
}


def evaluate_models(experiment:str, model_outputs:List[ModelOutput], stimuli:Stimuli, human_responses: HumanResponses, out_path:str='./results'):
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
    for model_output in model_outputs:
        model_responses = get_model_responses(model_output, stimuli)
        name = model_output.model_name
        print(name)
        accuracy[name], n[name] = compute_model_accuracy(model_responses, factors)
    
    # 3) create and save plots
    # main experiment figure
    plot_accuracy(experiment, accuracy, n, factors, out_path, \
        figure_contents=FIGURE_CONTENTS[stimuli.experiment], extended_interactions=False)

    # appendix figure with all interactions (extended - models plotted separately)
    plot_accuracy(experiment, accuracy, n, factors, out_path, extended_interactions=True)
    
    # 4) compute statistics
    if experiment in ["experiment1", "experiment2"]:
        calculate_anovas(accuracy, n, factors, WITHIN_FACTORS[experiment])
