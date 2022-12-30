# main models for figures 2 and 3
MAIN_MODELS = ['sort', 'deepsort', 'bytetrack', 'ocsort']

# additional models for figure 4
GATING_MODELS = ['deepsort', 'deepsort_nomotiongating', 'deepsort_nomotiongating_gtreid']

noisy_reid_values = [0.1, 0.825, 1.55, 2.275, 3.0]
NOISY_REID_MODELS = [f'deepsort_noisy_reid_{v}' for v in noisy_reid_values]
