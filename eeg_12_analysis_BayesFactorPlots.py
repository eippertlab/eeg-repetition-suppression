#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analysis - Bayes factor plots
-----------------------------
creates Bayes factor plots of EEG amplitudes, TFR power, SCR + pupil amplitudes
depending on settings for Figures 3, 4, S1-S4
Bayes facotrs from JASP need to be added manually as lists at the top of the script

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

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

fig3 = False
fig4 = False
figS1 = False
figS2 = False
figS3 = False
figS4 = True

save_type = 'svg'    # enter data format to which figures should be saved (e.g. svg, png), enter None to suppress saving
figure_path = '/data/pt_02299/ExpSupp/Main/results/figures/'

color_greenblue = (3/255, 121/255, 122/255)
color_redpurple = (175/255, 56/255, 94/255)
new_rc_params = {"font.family": 'Arial', "font.size": 6, "font.serif": [], "svg.fonttype": 'none'}
mpl.rcParams.update(new_rc_params)

# Bayes Factor manually entered from JASP output
if fig3:
    curr_col = [color_redpurple]
    folder = 'Fig3'
    BFs = pd.DataFrame({'Measure':['N1', 'N2P2', 'alpha', 'beta', 'gamma'], 'BF':[0.816, 4591, 0.183, 0.061, 1.655]})
elif fig4:
    curr_col = [color_redpurple]
    folder = 'Fig4'
    BFs = pd.DataFrame({'Measure':['N1', 'N2P2', 'alpha', 'beta', 'gamma', 'SCR', 'pupil'], 'BF':[0.275, 0.123, 0.447, 0.563, 0.601, 0.176, 0.095]})
elif figS1:
    curr_col = [color_greenblue]
    folder = 'FigS1'
    BFs = pd.DataFrame({'Measure':['N1', 'N2P2', 'alpha', 'beta', 'gamma', 'SCR', 'pupil'], 'BF':[0.207, 0.272, 0.188, 0.281, 0.225, 0.108, 0.519]})
elif figS2:
    curr_col = [color_redpurple]
    folder = 'FigS2'
    BFs = pd.DataFrame({'Measure':['N1', 'N2P2', 'alpha', 'beta', 'gamma'], 'BF':[0.365, 107000, 0.062, 0.075, 0.608]})
elif figS3:
    curr_col = [color_redpurple]
    folder = 'FigS3'
    BFs = pd.DataFrame({'Measure':['N1', 'N2P2', 'alpha', 'beta', 'gamma', 'SCR', 'pupil'], 'BF':[0.189, 0.200, 1.898, 1.957, 0.373, 0.106, 0.066]})
elif figS4:
    curr_col = [color_greenblue]
    folder = 'FigS4'
    BFs = pd.DataFrame({'Measure':['N1', 'N2P2', 'alpha', 'beta', 'gamma', 'SCR', 'pupil'], 'BF':[0.735, 0.188, 0.185, 0.191, 0.195, 0.070, 0.251]})

BFs_transformed = BFs
BFs_transformed['BF'] = BFs_transformed['BF']-1 # this is necessary to move the zero line to 1, which represents equal support for both hypothesis

fig, ax = plt.subplots(figsize=(2,4/3))
plt.barh(y='Measure', width='BF', left=1, data=BFs_transformed, log=True, color = curr_col)
plt.xticks([1/100, 1/10, 1/3, 1, 3, 10, 100], ['1/100', '1/10', '1/3', '1', '3', '10', '100'])
plt.xlim(1/100, 100)
plt.xlabel('Bayes factor')
plt.axvspan(1/100, 1/10, facecolor='0', alpha=0.2, zorder=0)
plt.axvspan(1/10, 1/3, facecolor='0.2', alpha=0.2, zorder=0)
plt.axvspan(1/3, 3, facecolor='0.5', alpha=0.2, zorder=0)
plt.axvspan(3, 10, facecolor='0.2', alpha=0.2, zorder=0)
plt.axvspan(10, 100, facecolor='0', alpha=0.2, zorder=0)
plt.gca().invert_yaxis()
if save_type:
    plt.savefig(figure_path + folder + '/BFs.' + save_type)
plt.show(block=True)
