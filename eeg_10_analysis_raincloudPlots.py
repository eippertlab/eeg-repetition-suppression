#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analysis of eeg data - raincloud amplitude plots
------------------------------------------------
creates raincloud plots of EEG amplitudes and TFR power
depending on settings for Figures 3, 4, S1-S4

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

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

data_path = '/data/pt_02299/ExpSupp/Main/results/eeg/'
figure_path = '/data/pt_02299/ExpSupp/Main/results/figures/'

task = 'expsupp'
first_4_runs = False    #set true for Fig.3,4,S1 and false for Fig.S2,S3,S4

timedomain = True
timefrequencydomain = False

create_fig3_S2 = True
create_fig4_S3 = True
create_figS1_S4 = True

save_figs = False

np.random.seed(1990)

# plot settings
color_green = (1/255, 129/255, 68/255)
color_blue = (5/255, 112/255, 176/255)
color_greenblue = (3/255, 121/255, 122/255)
color_red = (215/255, 48/255, 31/255)
color_purple = (136/255, 65/255, 157/255)
color_redpurple = (175/255, 56/255, 94/255)
new_rc_params = {"font.family": 'Arial', "font.size": 6, "font.serif": [], "svg.fonttype": 'none'}
mpl.rcParams.update(new_rc_params)
jitter = 0.1
figsize = (4/3,4/3)


# read data (if file does not exist, run 08_analysis_timedomain.py -> get_amplitudes, or matlab-scripts for the relevant figures (11, 13)
if timedomain:
    if first_4_runs:
        amps = pd.read_csv(data_path + 'amplitudes_N2P2_blocks1-4.csv')
    else:
        amps = pd.read_csv(data_path + 'amplitudes_N2P2_blocks1-8.csv')

    amps['N1_reduction_exp'] = amps['2nd_N1_rep_exp']-amps['1st_N1_rep_exp'] #2nd-1st to take care of negative amplitude -> reduction is positive
    amps['N1_reduction_une'] = amps['2nd_N1_rep_une']-amps['1st_N1_rep_une'] #2nd-1st to take care of negative amplitude -> reduction is positive
    amps['N2P2_reduction_exp'] = amps['1st_N2P2_rep_exp']-amps['2nd_N2P2_rep_exp']
    amps['N2P2_reduction_une'] = amps['1st_N2P2_rep_une']-amps['2nd_N2P2_rep_une']

if timefrequencydomain:
    if create_fig3_S2:
        if first_4_runs:
            pow_fig3 = pd.read_csv(data_path + 'fig3_tfr_rois.csv')
        else:
            pow_fig3 = pd.read_csv(data_path + 'figS2_tfr_rois.csv')
    if create_fig4_S3:
        if first_4_runs:
            pow_fig4 = pd.read_csv(data_path + 'fig4_tfr_rois.csv')
        else:
            pow_fig4 = pd.read_csv(data_path + 'figS3_tfr_rois.csv')
        pow_fig4['alpha_reduction_exp'] = pow_fig4['alpha2_rep_exp'] - pow_fig4['alpha1_rep_exp']  # 2nd-1st to take care of negative amplitude -> reduction is positive
        pow_fig4['alpha_reduction_une'] = pow_fig4['alpha2_rep_une'] - pow_fig4['alpha1_rep_une']  # 2nd-1st to take care of negative amplitude -> reduction is positive
        pow_fig4['beta_reduction_exp'] = pow_fig4['beta2_rep_exp'] - pow_fig4['beta1_rep_exp']  # 2nd-1st to take care of negative amplitude -> reduction is positive
        pow_fig4['beta_reduction_une'] = pow_fig4['beta2_rep_une'] - pow_fig4['beta1_rep_une']  # 2nd-1st to take care of negative amplitude -> reduction is positive
        pow_fig4['gamma_reduction_exp'] = pow_fig4['gamma1_rep_exp'] - pow_fig4['gamma2_rep_exp']
        pow_fig4['gamma_reduction_une'] = pow_fig4['gamma1_rep_une'] - pow_fig4['gamma2_rep_une']
    if create_figS1_S4:
        if first_4_runs:
            pow_figS1 = pd.read_csv(data_path + 'figS1_tfr_rois.csv')
        else:
            pow_figS1 = pd.read_csv(data_path + 'figS4_tfr_rois.csv')

def raincloud_plot(data, jitter, figsize, figtitle, colors, xticklabels, ylabel, filename):
    df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=(data.values.shape[0], 2)),
                               columns=['left', 'right'])
    df_x_jitter += [1, 3]
    fig1, ax1 = plt.subplots(1, 1, figsize=figsize, sharey=True)
    fig1.suptitle(figtitle)

    v1 = ax1.violinplot(data.iloc[:,0].values, positions=[0.5], showmeans=False, showextrema=False,
                        showmedians=False,
                        side="low", widths=1.2)  # `side` param requires matplotlib 3.9+
    for pc in v1['bodies']:
        pc.set_facecolor(colors[0])
        pc.set_alpha(0.7)

    v2 = ax1.violinplot(data.iloc[:,1], positions=[3.5], showmeans=False,
                        showextrema=False, showmedians=False,
                        side="high", widths=1.2)
    for pc in v2['bodies']:
        pc.set_facecolor(colors[1])
        pc.set_alpha(0.7)

    df_left = pd.DataFrame({'jitter': df_x_jitter['left'], 'amp': data.iloc[:,0]})
    df_right = pd.DataFrame({'jitter': df_x_jitter['right'], 'amp': data.iloc[:,1]})
    sns.scatterplot(ax=ax1, data=df_left, x='jitter', y='amp', color=colors[0], zorder=100, edgecolor="black", s=15)
    sns.scatterplot(ax=ax1, data=df_right, x='jitter', y='amp', color=colors[1], zorder=100, edgecolor="black", s=15)
    ax1.set_xticks([0.5, 3.5])
    ax1.set_xticklabels(xticklabels)
    ax1.tick_params(axis='x', which='major', labelsize=6)
    ax1.set_xlim(-0.5, 4.5)
    sns.despine()
    ax1.set_xlabel('')
    ax1.set_ylabel(ylabel)
    plt.subplots_adjust(left=0.15)
    for idx in data.index:
        ax1.plot(df_x_jitter.loc[idx, ['left', 'right']],
                 data.iloc[idx, [0, 1]],
                 color='grey', linewidth=0.3, linestyle='--', zorder=-1)
    if filename:
        plt.savefig(filename)
    plt.show(block=True)

if create_fig3_S2:
    if timedomain:
        if save_figs:
            if first_4_runs:
                out_file = figure_path + 'Fig3/N1_raincloud.svg'
            else:
                out_file = figure_path + 'FigS2/N1_raincloud.svg'
        else:
            out_file = None
        raincloud_plot(data=amps[['1st_N1_rep', '2nd_N1_rep']], jitter=jitter, figsize=figsize,
                   figtitle='N1 repetition suppression', colors=[color_redpurple, color_redpurple],
                   xticklabels=['1st stimulus', '2nd stimulus'], ylabel=r'Amplitude (µV)', filename=out_file)

        if save_figs:
            if first_4_runs:
                out_file = figure_path + 'Fig3/N2P2_raincloud.svg'
            else:
                out_file = figure_path + 'FigS2/N2P2_raincloud.svg'
        else:
            out_file = None
        raincloud_plot(data=amps[['1st_N2P2_rep', '2nd_N2P2_rep']], jitter=jitter, figsize=figsize,
                   figtitle='N2P2 repetition suppression', colors=[color_redpurple, color_redpurple],
                   xticklabels=['1st stimulus', '2nd stimulus'], ylabel=r'Amplitude (µV)', filename=out_file)

    if timefrequencydomain:
        if save_figs:
            if first_4_runs:
                out_file = figure_path + 'Fig3/alpha_raincloud.svg'
            else:
                out_file = figure_path + 'FigS2/alpha_raincloud.svg'
        else:
            out_file = None
        raincloud_plot(data=pow_fig3[['alpha1', 'alpha2']], jitter=jitter, figsize=figsize,
                       figtitle='alpha power repetition suppression', colors=[color_redpurple, color_redpurple],
                       xticklabels=['1st stimulus', '2nd stimulus'], ylabel=r'Power', filename=out_file)

        if save_figs:
            if first_4_runs:
                out_file = figure_path + 'Fig3/beta_raincloud.svg'
            else:
                out_file = figure_path + 'FigS2/beta_raincloud.svg'
        else:
            out_file = None
        raincloud_plot(data=pow_fig3[['beta1', 'beta2']], jitter=jitter, figsize=figsize,
                       figtitle='beta power repetition suppression', colors=[color_redpurple, color_redpurple],
                       xticklabels=['1st stimulus', '2nd stimulus'], ylabel=r'Power', filename=out_file)

        if save_figs:
            if first_4_runs:
                out_file = figure_path + 'Fig3/gamma_raincloud.svg'
            else:
                out_file = figure_path + 'FigS2/gamma_raincloud.svg'
        else:
            out_file = None
        raincloud_plot(data=pow_fig3[['gamma1', 'gamma2']], jitter=jitter, figsize=figsize,
                       figtitle='gamma power repetition suppression', colors=[color_redpurple, color_redpurple],
                       xticklabels=['1st stimulus', '2nd stimulus'], ylabel=r'Power', filename=out_file)

if create_fig4_S3:
    if timedomain:
        if save_figs:
            if first_4_runs:
                out_file = figure_path + 'Fig4/N1_raincloud.svg'
            else:
                out_file = figure_path + 'FigS3/N1_raincloud.svg'
        else:
            out_file = None
        raincloud_plot(data=amps[['N1_reduction_exp', 'N1_reduction_une']], jitter=jitter, figsize=figsize,
                       figtitle='N1 amplitude reduction, repetition trials', colors=[color_red, color_purple],
                       xticklabels=['expected', 'unexpected'], ylabel=r'Amplitude (µV)', filename=out_file)

        if save_figs:
            if first_4_runs:
                out_file = figure_path + 'Fig4/N2P2_raincloud.svg'
            else:
                out_file = figure_path + 'FigS3/N2P2_raincloud.svg'
        else:
            out_file = None
        raincloud_plot(data=amps[['N2P2_reduction_exp', 'N2P2_reduction_une']], jitter=jitter, figsize=figsize,
                       figtitle='N2P2 amplitude reduction, repetition trials', colors=[color_red, color_purple],
                       xticklabels=['expected', 'unexpected'], ylabel=r'Amplitude (µV)', filename=out_file)

    if timefrequencydomain:
        if save_figs:
            if first_4_runs:
                out_file = figure_path + 'Fig4/alpha_raincloud.svg'
            else:
                out_file = figure_path + 'FigS3/alpha_raincloud.svg'
        else:
            out_file = None
        raincloud_plot(data=pow_fig4[['alpha_reduction_exp', 'alpha_reduction_une']], jitter=jitter, figsize=figsize,
                       figtitle='alpha power, repetition trials', colors=[color_red, color_purple],
                       xticklabels=['expected', 'unexpected'], ylabel=r'Power', filename=out_file)

        if save_figs:
            if first_4_runs:
                out_file = figure_path + 'Fig4/beta_raincloud.svg'
            else:
                out_file = figure_path + 'FigS3/beta_raincloud.svg'
        else:
            out_file = None
        raincloud_plot(data=pow_fig4[['beta_reduction_exp', 'beta_reduction_une']], jitter=jitter, figsize=figsize,
                       figtitle='beta power, repetition trials', colors=[color_red, color_purple],
                       xticklabels=['expected', 'unexpected'], ylabel=r'Power', filename=out_file)

        if save_figs:
            if first_4_runs:
                out_file = figure_path + 'Fig4/gamma_raincloud.svg'
            else:
                out_file = figure_path + 'FigS3/gamma_raincloud.svg'
        else:
            out_file = None
        raincloud_plot(data=pow_fig4[['gamma_reduction_exp', 'gamma_reduction_une']], jitter=jitter, figsize=figsize,
                       figtitle='gamma power, repetition trials', colors=[color_red, color_purple],
                       xticklabels=['expected', 'unexpected'], ylabel=r'Power', filename=out_file)

if create_figS1_S4:
    if timedomain:
        if save_figs:
            if first_4_runs:
                out_file = figure_path + 'FigS1/N1_raincloud.svg'
            else:
                out_file = figure_path + 'FigS4/N1_raincloud.svg'
        else:
            out_file = None
        raincloud_plot(data=amps[['2nd_N1_omi_exp', '2nd_N1_omi_une']], jitter=jitter, figsize=figsize,
                       figtitle='N1, omission trials', colors=[color_green, color_blue],
                       xticklabels=['expected', 'unexpected'], ylabel=r'Amplitude (µV)', filename=out_file)

        if save_figs:
            if first_4_runs:
                out_file = figure_path + 'FigS1/N2P2_raincloud.svg'
            else:
                out_file = figure_path + 'FigS4/N2P2_raincloud.svg'
        else:
            out_file = None
        raincloud_plot(data=amps[['2nd_N2P2_omi_exp', '2nd_N2P2_omi_une']], jitter=jitter, figsize=figsize,
                       figtitle='N2P2, omission trials', colors=[color_green, color_blue],
                       xticklabels=['expected', 'unexpected'], ylabel=r'Amplitude (µV)', filename=out_file)

    if timefrequencydomain:
        if save_figs:
            if first_4_runs:
                out_file = figure_path + 'FigS1/alpha_raincloud.svg'
            else:
                out_file = figure_path + 'FigS4/alpha_raincloud.svg'
        else:
            out_file = None
        raincloud_plot(data=pow_figS1[['alpha2_omi_exp', 'alpha2_omi_une']], jitter=jitter, figsize=figsize,
                       figtitle='alpha power, omission trials', colors=[color_green, color_blue],
                       xticklabels=['expected', 'unexpected'], ylabel=r'Power', filename=out_file)

        if save_figs:
            if first_4_runs:
                out_file = figure_path + 'FigS1/beta_raincloud.svg'
            else:
                out_file = figure_path + 'FigS4/beta_raincloud.svg'
        else:
            out_file = None
        raincloud_plot(data=pow_figS1[['beta2_omi_exp', 'beta2_omi_une']], jitter=jitter, figsize=figsize,
                       figtitle='beta power, omission trials', colors=[color_green, color_blue],
                       xticklabels=['expected', 'unexpected'], ylabel=r'Power', filename=out_file)

        if save_figs:
            if first_4_runs:
                out_file = figure_path + 'FigS1/gamma_raincloud.svg'
            else:
                out_file = figure_path + 'FigS4/gamma_raincloud.svg'
        else:
            out_file = None
        raincloud_plot(data=pow_figS1[['gamma2_omi_exp', 'gamma2_omi_une']], jitter=jitter, figsize=figsize,
                       figtitle='gamma power, omission trials', colors=[color_green, color_blue],
                       xticklabels=['expected', 'unexpected'], ylabel=r'Power', filename=out_file)

