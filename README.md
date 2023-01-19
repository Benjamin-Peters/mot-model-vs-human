
# mot-model-vs-human

This repository contains the code to reproduce the results and figures from:

Peters, Butkus, & Kriegeskorte (2022). How do humans and machine learning models track multiple objects through occlusion? *SVRHM 2022 Workshop @ NeurIPS*. [[link]](https://openreview.net/pdf?id=n5OTU5qcj05)


[Abstract](#abstract) | [Getting Started](#getting-started) | [Data and Stimuli](#data-and-stimuli) | [Example trajectories](#example-trajectories) | [Citation](#citation)


## Abstract

Interacting with a complex environment often requires us to track multiple task-relevant objects not all of which are continually visible. The cognitive literature has focused on tracking a subset of visible identical abstract objects (e.g., circles), isolating the tracking component from its context in real-world experience. In the real world, object tracking is harder in that objects may not be continually visible and easier in that objects differ in appearance and so their recognition can rely on both remembered position and current appearance. Here we introduce a generalized task that combines tracking and recognition of valued objects that move in complex trajectories and frequently disappear behind occluders. Humans and models (from the computer-vision literature on object tracking) performed tasks varying widely in terms of the number of objects to be tracked, the number of distractors, the presence of an occluder, and the appearance similarity between targets and distractors. We replicated results from the human literature, including a deterioration of tracking performance with the number and similarity of targets and distractors. In addition, we find that increasing levels of occlusion reduce performance. All models tested here behaved in qualitatively different ways from human observers, showing superhuman performance for large numbers of targets, and subhuman performance under conditions of occlusion. Our framework will enable future studies to connect the human behavioral and engineering literatures, so as to test image-computable multiple-object-tracking models as models of human performance and to investigate how tracking and recognition interact under natural conditions of dynamic motion and occlusion. 

## Getting started

- Clone the directory: `git clone git@github.com:Benjamin-Peters/mot-model-vs-human.git` 
- navigate to the cloned directory and install requirements: `pip install -r requirements.txt`
- run `python main.py`. All data and stimuli will be automatically downloaded.

## Data and Stimuli

Human behavioral data, model behavior, and stimuli (i.e., videos) are stored at https://osf.io/n7mah. The data will be automatically downloaded when running `main.py`.


### Stimuli
Stimuli for the two experiments are stored in the [format of the MOT benchmark](https://github.com/JonathonLuiten/TrackEval/blob/master/docs/MOTChallenge-Official/Readme.md).

```
|—— annotations
    |—— train_cocoformat.json
    |—— train_detections.pkl
|—— train
    |—— <TrialName01>
        |—— det
            |—— det.txt
        |—— gt
            |—— gt.txt
        |—— img1
            |—— 000001.png
            |—— 000002.png
            |—— ...
        |—— seqinfo.ini
    |—— <TrialName02>
        |—— ...
|—— experimental_sessions
    |—— <SessionName01>.json
```
- `det.txt` and `gt.txt` contain detections and ground truth, respectively.
- `img1` contains the sequence of frames.
- `<SessionName01>.json` contains full descriptions of all trials in the format of the FlyingObjects toolbox (to be released).

### Human responses
- Human responses are stored in one csv file per experiment. Each row corresponds to a unique user (different participants are identified by different `user_session_id`) performing a trial (trials have unique names: `unique_bluprint_id`). 
- experimental factors are coded in the following columns:
    - `n_valued_objects`: number of target objects (2, 3, 4) 
    - `occlusion_levels`: degree of occlusion as a ratio of the occluded area (0.0, 0.2, 0.4) 
    - `category_similarity`: whether objects were sampled from different categories (low, 0) or from the same category (high, 1)
    - `n_distractor_objects`: number of distractor objects (2, 3, 4, 10)
- columns `response_0`, `response_1`, ... contain the name of the object which participants clicked on for their first, second, ... response.
- columns `rt_0`, `rt_1`, ... contain the corresponding reaction times (starting from the end of the motion period).

### Model outputs

Model outputs for SORT, DeepSORT, OC-SORT, and ByteTrack are detections assigned to track ids. All detections across all frames and trials are concatenated into one list (stored in field `track_bboxes` of the pickled dict). Correspondence of list elements to frames and trials can be established via the annotations file in the stimulus set.

When running `main.py`, these model outputs are transformed into 'model responses' in the same format as human responses as detailed in the paper. Model responses are then stored in csv files akin to human responses.

#### Additional model analyses

- As detailed in the supplement of the paper (i.e. Fig. 7), we removed the spatial gating from DeepSORT and evaluated the model output on Experiment 1. The model behavior is stored in `model_outputs/experiment1_gating/deepsort_nomotiongating-experiment1_gating.pkl`. 
- As a sanity check we also evaluated the model with perfect appearance model (i.e., reid model), which lead to perfect performance on in all conditions. (`model_outputs/experiment1_gating/deepsort_nomotiongating_gtreid-experiment1_gating.pkl`)

## Example trajectories


<table border="0" cellspacing="0" cellpadding="0"  style="border-collapse: collapse; border: none;">
 <tr  style="border: none;">
    <td  style="border: none;"><img  width="150" height="150" src="https://github.com/Benjamin-Peters/mot-model-vs-human/blob/main/img/1.gif?raw=true"></td> 
    <td  style="border: none;"><img  width="150" height="150" src="https://github.com/Benjamin-Peters/mot-model-vs-human/blob/main/img/2.gif?raw=true"></td>
    <td  style="border: none;"><img  width="150" height="150" src="https://github.com/Benjamin-Peters/mot-model-vs-human/blob/main/img/3.gif?raw=true"></td>
 </tr >
</table>

## Citation

```
@inproceedings{
    peters2022how,
    title={How do humans and machine learning models track multiple objects through occlusion?},
    author={Benjamin Peters and Eivinas Butkus and Nikolaus Kriegeskorte},
    booktitle={SVRHM 2022 Workshop @ NeurIPS },
    year={2022},
    url={https://openreview.net/forum?id=n5OTU5qcj05}
}
```
