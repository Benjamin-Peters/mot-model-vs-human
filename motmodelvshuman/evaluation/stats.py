
import pandas as pd
import numpy as np

from statsmodels.stats.anova import AnovaRM


def anova(df_anova, within):
    print()
    print(AnovaRM(data=df_anova, depvar='condition_mean', 
              subject='subject_idx',
              within=within,
              aggregate_func='mean').fit())
    print()

def anovas(means, n, factors, withins, human=True):
    if human:
        print('\n\ncalculating anovas for human')
        ms = means['human']
        n_subjects = n['human']
    else:
        print('\n\ncalculating anovas for model')
        ms = np.mean([ms for name, ms in means.items() if name != 'human'], axis=1) # collapsing across models
        n_subjects = ms.shape[0]

    if len(factors) == 3: # experiment 1
        n_target_object_levels, n_occlusion_levels, n_category_similarity_levels = 3,3,2
        subject_idx = np.repeat(np.repeat(np.repeat(np.arange(ms.shape[0])[:, None, None, None], n_target_object_levels, axis=1), n_occlusion_levels, axis=2), n_category_similarity_levels, axis=3)
        target_idx = np.repeat(np.repeat(np.repeat(np.arange(ms.shape[1])[None, :, None, None], n_subjects, axis=0), n_occlusion_levels, axis=2), n_category_similarity_levels, axis=3)
        occlusion_idx = np.repeat(np.repeat(np.repeat(np.arange(ms.shape[2])[None, None, :, None], n_subjects, axis=0), n_target_object_levels, axis=1), n_category_similarity_levels, axis=3)
        category_similarity_idx = np.repeat(np.repeat(np.repeat(np.arange(ms.shape[3])[None, None, None, :], n_subjects, axis=0), n_target_object_levels, axis=1), n_occlusion_levels, axis=2)
        df_anova = pd.DataFrame(dict(subject_idx=subject_idx.flatten(), 
                           n_targets=target_idx.flatten(), 
                           occlusion=occlusion_idx.flatten(), 
                           category_similarity=category_similarity_idx.flatten(),
                           condition_mean=ms.flatten()))
    elif len(factors) == 4: # experiment 2
        n_target_object_levels, n_occlusion_levels, n_category_similarity_levels, n_distractors_levels  = 2,3,2,2
        subject_idx = np.repeat(np.repeat(np.repeat(np.repeat(np.arange(ms.shape[0])[:, None, None, None, None], n_target_object_levels, axis=1), n_occlusion_levels, axis=2), n_category_similarity_levels, axis=3), n_distractors_levels, axis=4)
        target_idx = np.repeat(np.repeat(np.repeat(np.repeat(np.arange(ms.shape[1])[None, :, None, None, None], n_subjects, axis=0), n_occlusion_levels, axis=2), n_category_similarity_levels, axis=3), n_distractors_levels, axis=4)
        occlusion_idx = np.repeat(np.repeat(np.repeat(np.repeat(np.arange(ms.shape[2])[None, None, :, None, None], n_subjects, axis=0), n_target_object_levels, axis=1), n_category_similarity_levels, axis=3), n_distractors_levels, axis=4)
        category_similarity_idx = np.repeat(np.repeat(np.repeat(np.repeat(np.arange(ms.shape[3])[None, None, None, :, None], n_subjects, axis=0), n_target_object_levels, axis=1), n_occlusion_levels, axis=2), n_distractors_levels, axis=4)
        distractor_idx = np.repeat(np.repeat(np.repeat(np.repeat(np.arange(ms.shape[4])[None, None, None, None, :], n_subjects, axis=0), n_target_object_levels, axis=1), n_occlusion_levels, axis=2), n_category_similarity_levels, axis=3)

        print(subject_idx.shape)
        print(target_idx.shape)
        print(occlusion_idx.shape)
        print(category_similarity_idx.shape)
        print(distractor_idx.shape)
        print(ms.shape)

        df_anova = pd.DataFrame(dict(subject_idx=subject_idx.flatten(), 
                           n_targets=target_idx.flatten(),
                           occlusion=occlusion_idx.flatten(), 
                           category_similarity=category_similarity_idx.flatten(),
                           n_distractors= distractor_idx.flatten(),
                           condition_mean=ms.flatten()))
    else:
        raise ValueError('factors should be either 3 or 4') 

    print(df_anova)
    
    for within in withins:
        anova(df_anova, within=within)



def calculate_anovas(means:dict, n:dict, factors:dict, withins:list):
    anovas(means, n, factors, withins, human=True)
    anovas(means, n, factors, withins, human=False)    

 

