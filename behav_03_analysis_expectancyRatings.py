#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analysis behavioural data - expectancy ratings
----------------------------------------------
load data
raincloud plot

authors:
--------
Lisa-Marie Pohle

contact:
--------
lmpohle@cbs.mpg.de

date:
-----
5th June 2025
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

source_path = '/data/pt_02299/ExpSupp/Main/source_data/'
result_path = '/data/pt_02299/ExpSupp/Main/results/beh/'
figure_path = '/data/pt_02299/ExpSupp/Main/results/figures/Fig2/'

subjects = ['es01', 'es03', 'es04', 'es06', 'es07', 'es08', 'es09', 'es10', 'es11', 'es12', 'es13', 'es15', 'es17',
            'es18', 'es19', 'es20', 'es21', 'es22', 'es23', 'es25', 'es26', 'es27', 'es28', 'es30', 'es31', 'es32',
            'es33', 'es34', 'es35', 'es40', 'es41', 'es42', 'es44', 'es45', 'es46', 'es47']

save_type = 'svg'    # enter data format to which figures should be saved (e.g. svg, png), enter None to suppress saving

# plot settings
space = 4
low_color = 'burlywood'
high_color = (190/255, 120/255, 55/255)
new_rc_params = {"font.family": 'Arial', "font.size": 6, "font.serif": [], "svg.fonttype": 'none'}
mpl.rcParams.update(new_rc_params)

# read data (if dataframe does not exist, run 02_analysis_painRatings.py beforehand) and restructure dataframe
df = pd.read_csv(os.path.join(result_path, 'ratings_jasp.csv'))
df = df.drop(columns=df.columns[3:8])

def raincloud_plot(data, data_all, jitter, figsize, figtitle, colors, xticklabels, ylabel, filename):
    df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=(data.values.shape[0], 2)),
                               columns=['left', 'right'])
    df_x_jitter += [1.5, 3.5]
    fig1, ax1 = plt.subplots(1, 1, figsize=figsize, sharey=True)
    fig1.suptitle(figtitle)

    df = pd.melt(data_all, id_vars='Subject', value_vars=['ExpRating_Low', 'ExpRating_high'], var_name='Condition',ignore_index=False).reset_index(drop=True)
    df.rename(columns={'value':'ExpRating'}, inplace=True)
    df = df.replace({'ExpRating_Low':'Low', 'ExpRating_high':'High'})
    sns.violinplot(x = 'Condition', y='ExpRating', hue = 'Condition', data=df, split=True, inner=None, bw=0.3,
                   palette={'High':high_color, 'Low':low_color}, legend=False, width=2.5, order=[None, 'Low', None, None, 'High', None], linewidth=0)

    df_left = pd.DataFrame({'jitter': df_x_jitter['left'], 'amp': data.iloc[:,0]})
    df_right = pd.DataFrame({'jitter': df_x_jitter['right'], 'amp': data.iloc[:,1]})
    sns.scatterplot(ax=ax1, data=df_left, x='jitter', y='amp', color=colors[0], zorder=100, edgecolor="black", s=15)
    sns.scatterplot(ax=ax1, data=df_right, x='jitter', y='amp', color=colors[1], zorder=100, edgecolor="black", s=15)
    ax1.set_xticks([1, 4])
    ax1.set_xticklabels(xticklabels)
    ax1.tick_params(axis='x', which='major', labelsize=6)
    ax1.set_xlim(-0.5, 5.5)
    sns.despine()
    ax1.set_xlabel('')
    ax1.set_ylabel(ylabel)
    ax1.set_ylim([0,100])
    ax1.set_yticks(np.arange(0,101,10))
    plt.subplots_adjust(left=0.15)
    for idx in data.index:
        ax1.plot(df_x_jitter.loc[idx, ['left', 'right']],
                 data.iloc[idx, [0, 1]],
                 color=high_color, linewidth=0.3, zorder=-1)
    ax1.axhline(25, color='black', linestyle=':', linewidth=1, zorder=-2)
    ax1.axhline(75, color='black', linestyle=':', linewidth=1, zorder=-2)

    handles = []
    handles.append(plt.axhline(25, c='black', label="correct ratios", linestyle=':', linewidth=1, zorder=0))
    plt.legend(handles=handles, loc='upper left', frameon=False)

    if filename:
        plt.savefig(filename)
    plt.show(block=True)


filename = figure_path + 'expectancy_raincloud_includedSubjects_large.' + save_type
raincloud_plot(data=df[['ExpRating_Low', 'ExpRating_high']], data_all=df, jitter=0.1, figsize=(4, 4),
               figtitle='Expectancy of repeated stimuli', colors=[low_color, high_color],
               xticklabels=['Low', 'High'], ylabel='Rating (% of repeated stimuli)', filename=filename)