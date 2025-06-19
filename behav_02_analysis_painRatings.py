#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analysis behavioural data - pain ratings
----------------------------------------
get blockwise pain  and expectancy ratings (expectancy for next script)
average over blocks per participant and condition (high/low, 1st/2nd stimulus)
boxplot with single datapoints of pain ratings per condition

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
import math
import matplotlib as mpl


source_path = '/data/pt_02299/ExpSupp/Main/source_data/'
result_path = '/data/pt_02299/ExpSupp/Main/results/beh/'
figure_path = '/data/pt_02299/ExpSupp/Main/results/figures/Fig2/'

subjects = ['es01', 'es03', 'es04', 'es06', 'es07', 'es08', 'es09', 'es10', 'es11', 'es12', 'es13', 'es15', 'es17',
            'es18', 'es19', 'es20', 'es21', 'es22', 'es23', 'es25', 'es26', 'es27', 'es28', 'es30', 'es31', 'es32',
            'es33', 'es34', 'es35', 'es40', 'es41', 'es42', 'es44', 'es45', 'es46', 'es47']

load_type = 'events'    # whether to load data for plotting directly from event files ('events') or pre-created dataframe ('df')
save_df = True          # whether to save the created dataframe of correct response rates
save_type = 'svg'       # enter data format to which figures should be saved (e.g. svg, png), enter None to suppress saving

# plot settings
low_color = 'burlywood'
high_color = (190/255, 120/255, 55/255)
new_rc_params = {"font.family": 'Arial', "font.size": 6, "font.serif": [], "svg.fonttype": 'none'}
mpl.rcParams.update(new_rc_params)

if load_type == 'events':
    # read in event files for all subjects
    events_all = pd.DataFrame([])
    for sub in subjects:
        event_files = glob.glob(source_path + sub + os.sep + 'beh' + os.sep + '*expsupp*.csv')
        laser_files = glob.glob(source_path + sub + os.sep + 'beh' + os.sep + '*expsupp*laser.csv')
        event_files = set(event_files) - set(laser_files)
        event_files = np.array(sorted(event_files))
        events_sub = pd.DataFrame([])
        # select first row of each dataset only, because we are only interested in block-wise ratings
        for e in range(len(event_files)):
            events = pd.read_csv(event_files[e])
            events = events.iloc[0:1,:]
            events = events.loc[:,['Cond_Block', 'Subject', 'Stim_temp','RatingS1','RatingS2','RatingExp']]
            events_sub = pd.concat([events_sub, events], ignore_index=True, sort=False)
        events_all = pd.concat([events_all, events_sub], ignore_index=True, sort=False)

    # create subsets for p(rep) = low and p(rep) = high blocks for easier handling
    events_high = events_all[events_all['Cond_Block'].apply(lambda x: 'HIGH' in x)]
    events_low = events_all[events_all['Cond_Block'].apply(lambda x: 'LOW' in x)]

    exp_low = []
    exp_high = []
    pain_1_high = []
    pain_2_high = []
    pain_1_low = []
    pain_2_low = []
    pain = []

    # calculate average ratings per participant for pain and expectancy ratings
    for sub in subjects:
        # expectancy ratings
        low = np.mean(events_low[events_low['Subject']==sub].loc[:,'RatingExp'])
        high = np.mean(events_high[events_high['Subject']==sub].loc[:,'RatingExp'])
        exp_low.append(low)
        exp_high.append(high)

        # pain ratings (1 vs 2 equals to first vs second stimulus of a trial)
        low1 = np.mean(events_low[events_low['Subject'] == sub].loc[:, 'RatingS1'])
        low2 = np.mean(events_low[events_low['Subject'] == sub].loc[:, 'RatingS2'])
        high1 = np.mean(events_high[events_high['Subject'] == sub].loc[:, 'RatingS1'])
        high2 = np.mean(events_high[events_high['Subject'] == sub].loc[:, 'RatingS2'])
        pain_1_low.append(low1)
        pain_2_low.append(low2)
        pain_1_high.append(high1)
        pain_2_high.append(high2)
        pain.append(np.mean([low1, low2, high1, high2]))

    df = pd.DataFrame({'Subject':subjects,'ExpRating_Low':exp_low,'ExpRating_high':exp_high,
                              'PainRating_1st_high':pain_1_high, 'PainRating_2nd_high':pain_2_high,
                              'PainRating_1st_low':pain_1_low, 'PainRating_2nd_low':pain_2_low, 'PainRating_average':pain})
    if save_df:
        df.to_csv(os.path.join(result_path, 'ratings_jasp.csv'), index=False)
        
elif load_type == 'df':
    df = pd.read_csv(os.path.join(result_path, 'ratings_jasp.csv'))
    
else:
    raise Exception('No data was loaded, please enter either "events" or "df" as load_type!')

# order data for easier plotting
data_low = [df.PainRating_1st_low,df.PainRating_2nd_low]
data_high = [df.PainRating_1st_high,df.PainRating_2nd_high]

# boxplot (Fig. 2b)
fig, ax = plt.subplots(figsize = (1.5,4/3))
bp_low = ax.boxplot(data_low, positions=np.array(range(len(data_low)))*2.0-0.4, sym='', widths=0.6,
                    patch_artist = True, zorder=3, showcaps=False)
bp_high = ax.boxplot(data_high, positions=np.array(range(len(data_high)))*2.0+0.4, sym='', widths=0.6,
                     patch_artist = True, zorder=3, showcaps=False)

# scatterplot (Fig. 2b)
X_1_low, X_2_low, X_1_high, X_2_high = [], [], [], []
for i in range(36):
    X_1_low.append(np.random.normal((np.array(range(len(data_low)))*2.0-0.4)[0], 0.1))
    X_2_low.append(np.random.normal((np.array(range(len(data_low)))*2.0-0.4)[1], 0.1))
    X_1_high.append(np.random.normal((np.array(range(len(data_low)))*2.0+0.4)[0], 0.1))
    X_2_high.append(np.random.normal((np.array(range(len(data_low)))*2.0+0.4)[1], 0.1))
plt.scatter(X_1_low, df.PainRating_1st_low, zorder=4, c='black', alpha=0.4, s=5)
plt.scatter(X_2_low, df.PainRating_2nd_low, zorder=4, c='black', alpha=0.4, s=5)
plt.scatter(X_1_high, df.PainRating_1st_high, zorder=4, c='black', alpha=0.4, s=5)
plt.scatter(X_2_high, df.PainRating_2nd_high, zorder=4, c='black', alpha=0.4, s=5)

# adjust plot (Fig. 2b)
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['boxes'], facecolor=color)
    plt.setp(bp['medians'], color='black')
set_box_color(bp_low, low_color)
set_box_color(bp_high, high_color)
ax.set_ylim(0,100)
ax.set_yticks(np.arange(0,101,10))
plt.legend([bp_low["boxes"][0],
            bp_high['boxes'][0],
            plt.axhline(50, c='black', linestyle=':', lw=1),
            plt.axhline(65, c='darkgrey', linestyle=':', lw=1)],
           ['low', 'high', 'pain threshold', 'aimed rating'], ncol=2, loc=9, frameon=False)
ticks = ['low', 'high']
plt.xticks(range(0, len(ticks) * 2, 2), ticks)
ax.axhline(50, c='black', linestyle=':', zorder=2, linewidth = 1)
ax.axhline(65, c='darkgrey', linestyle=':', zorder=2, linewidth = 1)
ax.set_xticklabels(['1st stimulus', '2nd stimulus'])
ax.set_ylabel('Rating')
ax.set_title('Pain ratings')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
if save_type:
    plt.savefig(figure_path + 'painRatings_includedSubjects.' + save_type)
plt.show()