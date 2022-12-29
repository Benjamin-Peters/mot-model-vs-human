import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

import itertools

FACTORS_TEXT = {
    'occlusion_levels' : ['no occlusion', 'low occlusion', 'high occlusion'],
    'category_similarity' : ['low category similarity', 'high category similarity'],
}

# LINESTYLES = ['solid', 'dashed', (0, (3, 1, 1, 1, 1, 1))]
LINESTYLES = [ 'dashed', (0, (3, 1, 1, 1, 1, 1)), 'solid']
MARKERS = ['*', 'o', 'X', 'D', '^', 's']
YLIMS = [0.25, 1.02]

MODEL_COLORS = plt.cm.get_cmap('Greys')(np.flipud(np.linspace(0.3, .8, 6)))

LABELS_FONTSIZE = 16

def plot_main(ax, means, factor_idx, factors, n, levels, xlabel, bar_width=0.12):
    # we want to keep dimension of the factor_idx
    dims = tuple([d for d in range(1, len(factors)+1) if d != factor_idx+1]) 

    # plot human
    if 'human' in means.keys():
        ms = np.mean(means['human'], axis=dims)
        x_shift = - bar_width/2*len(means.keys())
        ax.bar(np.arange(len(levels))+ x_shift, np.mean(ms, axis=0), bar_width*2, color='orange',
                yerr=np.std(ms, axis=0)/np.sqrt(n['human']), label=f'humans ({n["human"]} subjects)')

    # plot model
    i = 2
    for name in sorted(means.keys()):
        if name == 'human':
            continue
        # ms = means[name]
        ms = np.mean(means[name], axis=dims)
        x_shift = - bar_width/2*len(means.keys()) + i*bar_width
        x = np.arange(len(levels))+ x_shift
        if 'deepsort' in name:
            model_simple_name = name
        else:
            model_simple_name = name.split('_')[0]
        ax.bar(x, np.mean(ms, axis=0), bar_width, color=MODEL_COLORS[i-2],
                yerr=np.std(ms, axis=0)/np.sqrt(n[name]))
        ax.scatter(x, (YLIMS[0]+0.05)*np.ones(len(x)), marker=MARKERS[i-2], color='black', s=15, label=f'{model_simple_name} ({n[name]} runs)')
        i += 1 # counting to shift bars
    
    ax.set_xticks(np.arange(len(levels)))
    ax.set_xticklabels(levels, fontsize=LABELS_FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=LABELS_FONTSIZE)
    ax.set_ylim(YLIMS)
    ax.tick_params(axis='y', labelsize=13)
    
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -1), prop={"size": 12})


def fig_main(dataset_name, plot_dir, means, n, factors): #, colors):
    width_ratios = [len(levels) for levels in factors.values()]
    fig, axes = plt.subplots(1, len(factors), figsize=(4*(len(factors)), 3), gridspec_kw={'width_ratios': width_ratios})
    axes[0].set_ylabel('accuracy', fontsize=LABELS_FONTSIZE)

    for factor_idx, (factor, levels) in enumerate(factors.items()):
        plot_main(axes[factor_idx], means, factor_idx, factors, n, levels, factor)

    fig.suptitle(f'{dataset_name}', fontsize=19)
    plt.savefig(Path(plot_dir) / f'{dataset_name}_results_main.pdf', bbox_inches='tight')
    return


def fig_interaction(factor_1, factor_2,
                    dataset_name, interactions_dir, means, n, factors, colors, expanded=False):

    # fig, axes = plt.subplots(1, len(means), figsize=(4*len(means), 3))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    
    factor_idx_1 = list(factors.keys()).index(factor_1)
    factor_idx_2 = list(factors.keys()).index(factor_2)

    print('creating interaction figure for factors', factor_1, factor_2)

    dims_reduce = tuple([d for d in range(1,len(factors)+1) if d != factor_idx_1+1 and d != factor_idx_2+1])
    factor_2_idx_ms = 1 if factor_idx_2 < factor_idx_1 else 2 # we need to find which dimension color_dim corresponds to in the ms
    
    # plot humans
    if 'human' in means.keys():
        n_subjects = n['human']
        ms = np.mean(means['human'], axis=dims_reduce)
        axes[0].set_title(f'humans ({n_subjects} subjects)', fontsize=LABELS_FONTSIZE)
        for j, factor_2_level in enumerate(factors[factor_2]):
            x = np.arange(len(factors[factor_1]))
            y = np.mean(ms.take(j, axis=factor_2_idx_ms), axis=0)
            ystd = np.std(ms.take(j, axis=factor_2_idx_ms), axis=0)/np.sqrt(n['human'])
            if factor_2 in FACTORS_TEXT:
                label = FACTORS_TEXT[factor_2][j]
            else:
                label = f'{factor_2} {factor_2_level}'
            axes[0].errorbar(x=x, y=y, yerr=ystd, color=colors[factor_idx_2][j], #linestyle=LINESTYLES[j],
                            label=factor_2_level)
            axes[0].legend(title=factor_2)

    model_count = 0
    for i, (name, ms) in enumerate(means.items()):
        
        if name == 'human':
            continue

        ms = np.mean(ms, axis=dims_reduce)
        x = np.arange(len(factors[factor_1]), dtype=float)
        x_random_offset = np.random.normal(0, 0.025*len(x), size=len(x))
        x += x_random_offset

        for j, factor_2_level in enumerate(factors[factor_2]):
            y = np.mean(ms.take(j, axis=factor_2_idx_ms), axis=0)
            ystd = np.std(ms.take(j, axis=factor_2_idx_ms), axis=0)/np.sqrt(n[name])
            if factor_2 in FACTORS_TEXT:
                label = FACTORS_TEXT[factor_2][j]
            else:
                label = f'{factor_2} {factor_2_level}'
            model_simple_name = name.split('_')[0]
            axes[1].errorbar(x=x, y=y, yerr=ystd, color=colors[factor_idx_2][j],# linestyle=LINESTYLES[j],
                            marker=MARKERS[model_count], markersize=5,
                            label=f'{model_simple_name}')
        model_count += 1
    
    axes[1].set_title('models', fontsize=LABELS_FONTSIZE)

    
    for i in range(2):
        axes[i].set_xticks(np.arange(len(factors[factor_1])), )
        axes[i].set_xticklabels(factors[factor_1], fontsize=LABELS_FONTSIZE)
        axes[i].set_xlabel(factor_1, fontsize=LABELS_FONTSIZE)
        axes[i].set_ylim(YLIMS)
        axes[i].tick_params(axis='y', labelsize=13)
        
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        if i > 0:
            axes[i].set_yticks([])
            axes[i].spines['left'].set_visible(False)

    axes[0].set_ylabel('accuracy', fontsize=LABELS_FONTSIZE)
    fig.suptitle(f'{dataset_name} interaction between {factor_1} and {factor_2}', fontsize=10, y=1.05)
    plt.savefig(Path(interactions_dir) / f'{dataset_name}_interaction_{factor_1}_{factor_2}.pdf', bbox_inches='tight')


def figs_interactions(dataset_name, plot_dir, means, n, factors, colors, figure_contents=None):
    interactions_dir = Path(plot_dir) / 'interactions'
    interactions_dir.mkdir(parents=True, exist_ok=True)

    if figure_contents is None: # just doing all if not specified
        interactions = list(itertools.permutations(factors, 2))
    else:
        interactions = figure_contents['interactions']

    for interaction in interactions:
        fig_interaction(*interaction, dataset_name, interactions_dir, means, n, factors, colors)


def plot_accuracy(experiment:str, accuracy:dict, n:dict, factors:dict, out_path:str, figure_contents:dict=None):
    plot_dir = Path(out_path) / 'figures' / experiment
    plot_dir.mkdir(parents=True, exist_ok=True)

    cmap_names = ['Blues', 'Greens', 'Oranges', 'Purples', 'Reds']
    colors = {}

    # plotting main results in single dimensions of factors
    if figure_contents["main"] or figure_contents is None:
        fig_main(experiment, plot_dir, accuracy, n, factors) #, colors)
    
    for i, (factor, levels) in enumerate(factors.items()):
        colors[i] = plt.cm.get_cmap(cmap_names[i])(np.flipud(np.linspace(0.2, .8, len(levels))))
    figs_interactions(experiment, plot_dir, accuracy, n, factors, colors, figure_contents)


