import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment


def get_trajectories(
                    blueprint,                            
                    object_name: str = None,      
                    trajectory_type: str = 'position_trajectory',
                    target_str: str = 'Target',
                    distractor_str: str = 'Distractor',
                    occluder_str: str = 'Occluder'
                    ):

    objects = blueprint['objects']
    
    target_objects = [obj for obj in objects if target_str in obj['name']]
    distractor_objects = [obj for obj in objects if distractor_str in obj['name']]
    objects = target_objects + distractor_objects
    occluder_objects = [obj for obj in objects if occluder_str in obj['name']]    
    
    n_target_objects = len(target_objects)
    
    for obj in objects:
        name_parts = obj['name'].split()
        if name_parts[0] == 'Distractor':
            obj['adjusted_name'] = ' '.join([*name_parts[:2], str(int(name_parts[2])-n_target_objects)])
        else:
            obj['adjusted_name'] = obj['name']

            
    if object_name is not None:
        selected_objects = [obj for obj in objects if object_name in obj['adjusted_name']]
    else:
        selected_objects = target_objects + distractor_objects
    
    n_frames = len(objects[0][trajectory_type]['trajectory'])
    trajectories = np.zeros((n_frames, len(selected_objects), 3))
    object_names = []
    for i, obj in enumerate(selected_objects):
        trajectories[:, i, :] = np.array(obj[trajectory_type]['trajectory'])[:, :3]
        object_names.append(obj['adjusted_name'])
    return trajectories, object_names

def get_position_trajectories(
                    blueprint,                            
                    object_name: str = None,      
                    target_str: str = 'Target',
                    distractor_str: str = 'Distractor',
                    occluder_str: str = 'Occluder'):
    return get_trajectories(blueprint, object_name, 'position_trajectory', target_str, distractor_str, occluder_str)



def get_object_pos(model_detections: pd.DataFrame,
                   img_width:int = 512,
                   img_height:int = 512):
    n_objects = model_detections.shape[0]
    model_object_pos = np.zeros((n_objects, 2))
    i = 0
    for _, row in model_detections.iterrows():
        x = (row['bbox_left'] + row['bbox_w']/2)/img_width - 0.5
        y = (row['bbox_top'] + row['bbox_h']/2)/img_height - 0.5
        model_object_pos[i,:] = [x, -y]
        i += 1
    
    return model_object_pos

def get_closest_match(model_detections: pd.DataFrame,
                    target_pos: np.ndarray, 
                    distractor_pos: np.ndarray,
                    verbose: bool = True):
    """ returns the object ids of the tracker which correspond to closest match to the observations
    
        this assumes that the tracker can understand the cueing process perfectly

    Args:
        detections (pd.DataFrame): all detections of the tracker in this trial

    Returns:
        _type_: _description_
    """
    
    n_targets = target_pos.shape[0]
    n_distractors = distractor_pos.shape[0]
    object_names = [f'Target Object {i+1}' for i in range(n_targets)] + [f'Distractor Object {i+1}' for i in range(n_distractors)]

    gt_object_pos = np.concatenate((target_pos, distractor_pos), axis=0)
    model_object_pos = get_object_pos(model_detections) # converting from model bboxes to pos in blueprint xy space

    # get ids from the model for this frame
    model_ids = model_detections.obj_id.to_numpy(dtype=int)

    # distances shape: (n_objects, n_model_detections)
    distances = np.linalg.norm(gt_object_pos[:,None,:] - model_object_pos[None,:,:], axis=2)

    # assigning ground truth objects to model detections
    gt_object_idxs, model_object_idxs = linear_sum_assignment(distances)

    tracklets = []
    for gt_object_idx, model_object_idx in zip(gt_object_idxs, model_object_idxs):
        model_id = model_ids[model_object_idx] # converting assignment in current frame to global model id

        if gt_object_idx < n_targets:
            gt_type = 'target'
            gt_ind = gt_object_idx
        else:
            gt_type = 'distractor'
            gt_ind = gt_object_idx - n_targets

        object_name = object_names[gt_object_idx]
        tracklets.append({'model_id': model_id, 'gt_ind': gt_ind, 'gt_type': gt_type, 'object_name': object_name})

    return tracklets    



def simple_model_response_per_trial(
                        model_output_trial:pd.DataFrame,
                        target_pos_traj,
                        distractor_pos_traj,
                        first_frame: int, 
                        min_prop_objects_observed: float = 0.0,
                        verbose:bool=True):
    
    def printv(x):
        if verbose:
            print(x)

    n_target_objects = target_pos_traj.shape[1]
    n_distractor_objects = distractor_pos_traj.shape[1]

    last_frame = model_output_trial.frame_idx.max()

    printv(f"model output at frame {first_frame}:")
    printv(model_output_trial[model_output_trial.frame_idx == first_frame])

    printv('matching observations to ground truth objects on first frame (frame {})'.format(first_frame))
    start_tracklets = get_closest_match(model_output_trial[model_output_trial.frame_idx == first_frame],
                                target_pos_traj[first_frame-1], # traj has zero-based index, so we need to subtract 1
                                distractor_pos_traj[first_frame-1],
                                verbose=verbose)
    printv(f'start first_frame={first_frame}')
    for x in start_tracklets:
        printv(x)
    
    printv('matching observations to ground truth objects on last frame (frame {})'.format(last_frame))
    end_tracklets = get_closest_match(model_output_trial[model_output_trial.frame_idx == last_frame],
                                target_pos_traj[last_frame-1],
                                distractor_pos_traj[last_frame-1],
                                verbose=verbose)    
    printv(f'end last_frame={last_frame}')
    for x in end_tracklets:
        printv(x)

    responses = []
    is_guess = []
    #all_object_names = [f'Target Object {i}' for i in range(n_target_objects)] + [f'Distractor Object {i}' for i in range(n_target_objects, n_distractor_objects+n_target_objects)]
    all_object_names = [f'Target Object {i+1}' for i in range(n_target_objects)] + [f'Distractor Object {i+1}' for i in range(n_distractor_objects)]
    
    # give "responses" for succesfully tracked target objects
    for start_tracklet in start_tracklets:
        if start_tracklet['gt_type'] == 'target':
            for end_tracklet in end_tracklets:
                if end_tracklet['model_id'] == start_tracklet['model_id']:
                    #responses.append(f"Target Object {start_tracklet['gt_ind']}")
                    responses.append(end_tracklet['object_name']) # this could be a target or distractor at the end
                    is_guess.append(False)

    # the remaining responses are guesses
    n_responses_missing = n_target_objects - len(responses)
    for i in range(n_responses_missing):
        remaining_objects = [x for x in all_object_names if x not in responses]
        responses.append(remaining_objects[np.random.randint(low=0, high=len(remaining_objects))])
        is_guess.append(True)
    
    printv(f'responses={responses}')
    printv(f'is_guess={is_guess}')
    
    return dict(responses=responses, is_guess=is_guess)


def get_blueprint(blueprints, bpid):
    for bp in blueprints:
        if bp['unique_blueprint_id'] == bpid:
            return bp
    return None

def simple_response(model_output, 
                    blueprints,
                    n_runs:int=1, 
                    model_name:str='MODEL-NAME',
                    experimental_session_id:str='',
                    num_max_responses:int=4,
                    first_frame:int = 20,
                    min_prop_objects_observed: float = 0.0,
                    verbose:bool=False):

    if 'bpid' not in model_output.columns:
        raise Exception('\n\nwarning - there is currently no matching via bpid\n\n'.upper())

    model_bpids = model_output['bpid'].values

    n_incorrect = 0

    response_rows = []
    for run in range(n_runs):
        for (trial_idx, bpid) in enumerate(model_output['bpid'].unique()):

            # getting the blueprint according to the bpid from the model output
            blueprint = get_blueprint(blueprints, bpid)
            if blueprint is None:
                raise Exception(f'WARNING: no blueprint found for bpid {bpid}')

            # getting the trajectories of targets and distractors from the blueprint
            pos_traj, names = get_position_trajectories(blueprint)
            target_idxs =  [name.startswith("Target") for name in names]
            distractor_idxs =  [name.startswith("Distractor") for name in names]
            target_pos_traj = pos_traj[:, target_idxs , :2]
            distractor_pos_traj = pos_traj[:, distractor_idxs, :2]

            # selecting the model_output portion that matches the blueprint id
            model_output_trial = model_output[model_output.bpid == bpid]

            out_dict = simple_model_response_per_trial(
                                model_output_trial=model_output_trial,
                                target_pos_traj=target_pos_traj,
                                distractor_pos_traj=distractor_pos_traj,
                                first_frame=first_frame,
                                min_prop_objects_observed = min_prop_objects_observed, 
                                verbose=verbose)

            responses = out_dict['responses']
            is_guess = out_dict['is_guess']

            if len(responses) > num_max_responses:
                raise ValueError(f"There are {len(responses)} responses in trial {trial_idx} but the maximum number of responses is {num_max_responses} - you should set num_max_responses to a higher value")
                        
            def get_resp(idx):
                if idx <= len(responses)-1:
                    return responses[idx]
                else:
                    return None
                
            def get_correct(idx):
                resp = get_resp(idx)
                if resp is None:
                    return None
                else:
                    return resp.startswith("Target")
                
            def get_is_guess(idx):
                if idx <= len(responses)-1:
                    return is_guess[idx]
                else:
                    return None                
            
            # build up new row in the response dataframe
            row = dict(
                unique_blueprint_id=bpid,
                trial_idx=trial_idx)            
            
            for i in range(num_max_responses):
                row[f'response_{i}'] = get_resp(i)
                row[f'correct_{i}'] = get_correct(i)
                row[f'is_guess_{i}'] = get_is_guess(i)

                if row[f'correct_{i}'] is False:
                    n_incorrect += 1

            row.update(dict(                
                            user_session_id=f'{model_name}-simple-run-{run:02d}',
                            run=run,
                            model_name=f'{model_name}-simple',
                            experimental_session_id=experimental_session_id)
                       )
            
            # append row to rows list
            response_rows.append(row)
            
            if 'meta_information' in blueprint:
                response_rows[-1].update(blueprint['meta_information'])
    
    responses = pd.DataFrame(response_rows)
        
    return responses