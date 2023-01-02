import itertools
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


FACTORS_TEXT = {
    'occlusion_levels' : ['no occlusion', 'low occlusion', 'high occlusion'],
    'category_similarity' : ['low category similarity', 'high category similarity'],
}

# LINESTYLES = ['solid', 'dashed', (0, (3, 1, 1, 1, 1, 1))]
LINESTYLES = [ 'dashed', (0, (3, 1, 1, 1, 1, 1)), 'solid']
MARKERS = ['*', 'o', 'X', 'D', '^', 's', 'v', 'p']
YLIMS = [0.25, 1.02]

MODEL_COLORS = plt.cm.get_cmap('Greys')(np.flipud(np.linspace(0.3, .8, 6)))

LABELS_FONTSIZE = 16
MODEL_NAMES_FONTSIZE = 10

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
        model_simple_name = name.split('-')[0]
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


def fig_interactions(interactions, experiment, interactions_dir, means, n, factors, colors, figure_contents=None, extended_interactions=False):
    print('plotting interactions', interactions)

    n_rows = len(interactions)
    n_cols = len(means) if extended_interactions else 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), gridspec_kw={'hspace': 0.5})

    for (i, (factor_1, factor_2)) in enumerate(interactions):
        factor_idx_1 = list(factors.keys()).index(factor_1)
        factor_idx_2 = list(factors.keys()).index(factor_2)

        dims_reduce = tuple([d for d in range(1,len(factors)+1) if d != factor_idx_1+1 and d != factor_idx_2+1])
        factor_2_idx_ms = 1 if factor_idx_2 < factor_idx_1 else 2 # we need to find which dimension color_dim corresponds to in the ms
    
        # plot humans
        if 'human' in means.keys():
            n_subjects = n['human']
            ms = np.mean(means['human'], axis=dims_reduce)
            ax = axes[i,0] if len(interactions) > 1 else axes[0]
            if i == 0:
                ax.set_title(f'humans ({n_subjects} subjects)', fontsize=MODEL_NAMES_FONTSIZE, fontweight="bold")
            for j, factor_2_level in enumerate(factors[factor_2]):
                x = np.arange(len(factors[factor_1]))
                y = np.mean(ms.take(j, axis=factor_2_idx_ms), axis=0)
                ystd = np.std(ms.take(j, axis=factor_2_idx_ms), axis=0)/np.sqrt(n['human'])
                if factor_2 in FACTORS_TEXT:
                    label = FACTORS_TEXT[factor_2][j]
                else:
                    label = f'{factor_2} {factor_2_level}'
                ax.errorbar(x=x, y=y, yerr=ystd, color=colors[factor_idx_2][j], #linestyle=LINESTYLES[j],
                                label=factor_2_level)
                ax.legend(title=factor_2)

        model_means = {k:v for k,v in means.items() if k != 'human'}
        for j, (model_name, ms) in enumerate(model_means.items()):
            ms = np.mean(ms, axis=dims_reduce)
            x = np.arange(len(factors[factor_1]), dtype=float)
            x_random_offset = np.random.normal(0, 0.025*len(x), size=len(x))
            x += x_random_offset

            for k, factor_2_level in enumerate(factors[factor_2]):
                y = np.mean(ms.take(k, axis=factor_2_idx_ms), axis=0)
                ystd = np.std(ms.take(k, axis=factor_2_idx_ms), axis=0)/np.sqrt(n[model_name])
                if factor_2 in FACTORS_TEXT:
                    label = FACTORS_TEXT[factor_2][k]
                else:
                    label = f'{factor_2} {factor_2_level}'
                model_simple_name = model_name.split('-')[0]
                if extended_interactions:
                    ax = axes[i,j+1] if len(interactions) > 1 else axes[j+1]
                else:
                    ax = axes[i,1] if len(interactions) > 1 else axes[1]
                ax.errorbar(x=x, y=y, yerr=ystd, color=colors[factor_idx_2][k],# linestyle=LINESTYLES[j],
                                marker=MARKERS[j+1], markersize=5,
                                label=f'{model_simple_name}')
                if i == 0:
                    name = 'models' if not extended_interactions else f'{model_simple_name}'
                    ax.set_title(name, fontsize=MODEL_NAMES_FONTSIZE, fontweight="bold")
            
        for j in range(n_cols):
            ax = axes[i,j] if n_rows > 1 else axes[j]
            ax.set_xticks(np.arange(len(factors[factor_1])), )
            ax.set_xticklabels(factors[factor_1], fontsize=LABELS_FONTSIZE)
            ax.set_xlabel(factor_1, fontsize=LABELS_FONTSIZE)
            ax.set_ylim(YLIMS)
            ax.tick_params(axis='y', labelsize=13)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if j > 0:
                ax.set_yticks([])
                ax.spines['left'].set_visible(False)

        ax = axes[i,0] if len(interactions) > 1 else axes[0]
        ax.set_ylabel('accuracy', fontsize=LABELS_FONTSIZE)

        #fig.suptitle(f'{experiment} interaction between {factor_1} and {factor_2}', fontsize=10, y=1.05)

        if n_rows == 1:
            plt.savefig(Path(interactions_dir) / f'{experiment}_interaction_{factor_1}_{factor_2}.pdf', bbox_inches='tight')

    if n_rows > 1:
        plt.savefig(Path(interactions_dir) / f'{experiment}_interactions.pdf', bbox_inches='tight')


def plot_accuracy(experiment:str, accuracy:dict, n:dict, factors:dict, out_path:str,
                    figure_contents:dict=None, extended_interactions:bool=False):
    plot_dir = Path(out_path) / 'figures' / experiment
    plot_dir.mkdir(parents=True, exist_ok=True)

    cmap_names = ['Blues', 'Greens', 'Oranges', 'Purples', 'Reds']
    colors = {}

    # plotting main results in single dimensions of factors
    fig_main(experiment, plot_dir, accuracy, n, factors) #, colors)
    
    for i, (factor, levels) in enumerate(factors.items()):
        colors[i] = plt.cm.get_cmap(cmap_names[i])(np.flipud(np.linspace(0.2, .8, len(levels))))

    extended_str = '_extended' if extended_interactions else ''
    interactions_dir = Path(plot_dir) / f'interactions{extended_str}'
    interactions_dir.mkdir(parents=True, exist_ok=True)
    if figure_contents is None: # just doing all if specific interactions are not specified
        interactions = list(itertools.permutations(factors, 2))
        fig_interactions(interactions, experiment, interactions_dir, accuracy, n, factors, colors, figure_contents, extended_interactions=True)
    else:
        for interaction in figure_contents['interactions']:
            fig_interactions([interaction], experiment, interactions_dir, accuracy, n, factors, colors, extended_interactions=False)


