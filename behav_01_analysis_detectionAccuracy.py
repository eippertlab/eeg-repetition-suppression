#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analysis behavioural data - detection accuracy
----------------------------------------------
load data and categorize responses as correct/wrong
calculate percentages of correct answers
boxplot accuracies

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
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib as mpl

source_path = '/data/pt_02299/ExpSupp/Main/raw_data/'
result_path = '/data/pt_02299/ExpSupp/Main/results/beh/'
figure_path = '/data/pt_02299/ExpSupp/Main/results/figures/Fig2/'

subjects = ['es01', 'es03', 'es04', 'es06', 'es07', 'es08', 'es09', 'es10', 'es11', 'es12', 'es13', 'es15', 'es17',
            'es18', 'es19', 'es20', 'es21', 'es22', 'es23', 'es25', 'es26', 'es27', 'es28', 'es30', 'es31', 'es32',
            'es33', 'es34', 'es35', 'es40', 'es41', 'es42', 'es44', 'es45', 'es46', 'es47']

load_type = 'events'   # whether to load data for plotting directly from event files ('events') or pre-created dataframe ('df')
save_df = True         # whether to save the created dataframe of correct response rates
save_type = 'svg'      # enter data format to which figures should be saved (e.g. svg, png), enter None to suppress saving

new_rc_params = {"font.family": 'Arial', "font.size": 6, "font.serif": [], "svg.fonttype": 'none'}
mpl.rcParams.update(new_rc_params)

def categorize(row):
    if np.isnan(row['decision']):
        return 'NaN'
    elif row['decision'] == 2.0:
        if row['Rep'] == 1:
            return 'correct'
        else:
            return 'false'
    elif row['decision'] == 1.0:
        if row['Rep'] == 0:
            return 'correct'
        else:
            return 'false'

if load_type == 'events':
    # read event files and categorize participants' decisions (correct, false, NaN)
    events_all = pd.DataFrame([])
    for sub in subjects:
        event_files = glob.glob(source_path + 'sub-' + sub + '/eeg/sub-' + sub + '_task-detection_run-*_events.tsv')
        event_files = np.array(sorted(event_files))
        events_sub = pd.DataFrame([])
        for e in range(len(event_files)):
            events = pd.read_csv(event_files[e])
            events_sub = pd.concat([events_sub, events], ignore_index=True, sort=False)
        events_sub['decision_category'] = events_sub.apply(lambda row: categorize(row), axis=1)
        events_all = pd.concat([events_all, events_sub], ignore_index=True, sort=False)

    # encoding of laser aborts changed between participants, make them equal (if first lasershot worked
    # (i.e. Lasershot1 == 1), but second not, psychopy tried to send a trigger for the second (i.e. Lasershot2 == -99 ->
    # trigger sent, but no stimulation), but for some participants Lasershot2 == 0, which means that not even a trigger is
    # present, thus we want to change this)
    events_all['Lasershot2'] = np.where((events_all.Lasershot1 == 1), np.where((events_all.Lasershot2 == 0), -99, events_all.Lasershot2), events_all.Lasershot2)

    # drop rows, where laser aborted depending on when this happened in a trial:
    # -99 = before first stimulus, remove only this trial
    # 0 = actually no abort, but due to errors in correctly detecting it as no abort during the experiment no decision task
    #     was shown, thus one needs to delete both this and the next trial from the rating to remove confusion (counting
    #     two trials as two stimuli within a trial)
    # 1 = during first stimulus, remove this trial if choice tak was presented (only if both Lasershot1 and Lasershot2
    #     are 1), else remove this and the following (see abort = 0 for rationale)
    # 2 = between first and second stimulus, see abort = 1
    # 3 = during second stimulus (only possible for repetition trials), remove this trial, but keep next since choice task
    #     was always shown
    # 4 = after second stimulus, irrelevant for choice task
    drop_idx = []
    for index, row in events_all.iterrows():
        if (row.aborted == -99) and (row.Lasershot1 == -99) and (row.Lasershot2 == 0):
            drop_idx.append(index)
        elif row.aborted == 0:
            if (row.Lasershot1 == 1) and (row.Lasershot2 == 1):
                continue
            elif (row.Lasershot1 == 1) and (row.Lasershot2 == -99):
                drop_idx.append(index)
                drop_idx.append(index+1)
            else:
                raise Exception('Unknown condition: aborted=0, Lasershot1=' + row.Lasershot1 + ', Lasershot2=' + row.Lasershot2 + 'in row ' + str(index))
        elif row.aborted == 1:
            if (row.Rep == 0) and (row.Lasershot1 == 1) and (row.Lasershot2 == 1):
                drop_idx.append(index)
            elif (row.Rep == 1) and (row.Lasershot1 == -99) and (row.Lasershot2 == 0):
                drop_idx.append(index)
                drop_idx.append(index + 1)
            elif (row.Rep == 1) and (row.Lasershot1 == 1) and (row.Lasershot2 == -99):
                drop_idx.append(index)
                drop_idx.append(index + 1)
            else:
                raise Exception('Unknown condition: aborted=1, Lasershot1=' + row.Lasershot1 + ', Lasershot2=' + row.Lasershot2 + 'in row ' + str(index))
        elif row.aborted == 2:
            if (row.Rep == 0) and (row.Lasershot1 == 1) and (row.Lasershot2 == 1):
                drop_idx.append(index)
            elif (row.Rep == 1) and (row.Lasershot1 == -99) and (row.Lasershot2 == 0):
                drop_idx.append(index)
                drop_idx.append(index+1)
            elif (row.Rep == 1) and (row.Lasershot1 == 1) and (row.Lasershot2 == -99):
                drop_idx.append(index)
                drop_idx.append(index + 1)
            else:
                raise Exception('Unknown condition: aborted=2, Lasershot1=' + row.Lasershot1 + ', Lasershot2=' + row.Lasershot2 + 'in row ' + str(index))
        elif row.aborted == 3:
            if (row.Rep == 1) and (row.Lasershot1 == 1) and (row.Lasershot2 == 1):
                drop_idx.append(index)
            else:
                raise Exception('Unknown condition: aborted=3, Lasershot1=' + row.Lasershot1 + ', Lasershot2=' + row.Lasershot2 + 'in row ' + str(index))
        elif row.aborted == 4:
            if (row.Lasershot1 == 1) and (row.Lasershot2 == 1):
                continue
            else:
                raise Exception('Unknown condition: aborted=4, Lasershot1=' + row.Lasershot1 + ', Lasershot2=' + row.Lasershot2 + 'in row ' + str(index))
        else:
            raise Exception('No abort specified, row index = ' + str(index))
    drop_idx = set(drop_idx)
    events_all = events_all.drop(drop_idx)

    # count correct, false and no answers per participant, block and category (one vs. two stimuli)
    df = events_all.groupby(['Subject', 'Run', 'Rep', 'decision_category']).size().reset_index().rename(columns={0: 'counter'})
    df = df.pivot(index=['Subject','Run', 'decision_category'], columns='Rep', values='counter')
    df = df.rename(columns={0: 'OneStim', 1: 'TwoStim'})

    # fill empty conditions with NaN and calculate overall accuracy (collapse over condition)
    insert = pd.DataFrame(index=[('es03', 2, 'NaN'), ('es07', 1, 'NaN'), ('es07', 2, 'NaN'), ('es07', 2, 'false'),
                                 ('es08', 1, 'NaN'), ('es09', 1, 'NaN'), ('es09', 2, 'NaN'), ('es09', 2, 'false'),
                                 ('es10', 2, 'NaN'), ('es11', 2, 'NaN'), ('es13', 2, 'NaN'), ('es13', 2, 'false'),
                                 ('es14', 1, 'NaN'), ('es14', 2, 'NaN'), ('es17', 1, 'NaN'), ('es17', 2, 'NaN'),
                                 ('es17', 2, 'false'), ('es18', 1, 'NaN'), ('es18', 2, 'NaN'), ('es19', 2, 'NaN'),
                                 ('es20', 2, 'NaN'), ('es25', 1, 'NaN'), ('es26', 1, 'NaN'), ('es26', 1, 'false'),
                                 ('es27', 1, 'NaN'), ('es27', 2, 'NaN'), ('es30', 1, 'NaN'), ('es30', 2, 'NaN'),
                                 ('es31', 1, 'NaN'), ('es31', 2, 'NaN'), ('es32', 1, 'NaN'), ('es32', 2, 'NaN'),
                                 ('es33', 2, 'NaN'), ('es35', 2, 'NaN'), ('es39', 2, 'NaN'), ('es40', 1, 'false'),
                                 ('es40', 2, 'NaN'), ('es42', 2, 'NaN'), ('es44', 2, 'NaN'), ('es45', 2, 'NaN'),
                                 ('es46', 2, 'NaN'), ('es05', 1, 'NaN'), ('es16', 1, 'NaN'), ('es16', 2, 'correct'),
                                 ('es16', 2, 'false'), ('es16', 2, 'NaN'), ('es24', 1, 'NaN'), ('es29', 2, 'correct'),
                                 ('es29', 2, 'false'), ('es29', 2, 'NaN'), ('es36', 2, 'correct'), ('es36', 2, 'false'),
                                 ('es36', 2, 'NaN'), ('es38', 1, 'NaN'), ('es38', 2, 'correct'), ('es38', 2, 'false'),
                                 ('es38', 2, 'NaN'), ('es43', 2, 'correct'), ('es43', 2, 'false'), ('es43', 2, 'NaN'),
                                 ('es47', 1, 'NaN'), ('es47', 2, 'false')],
                          data={'OneStim': 0, 'TwoStim': 0})
    df = pd.concat([df, insert], sort=False)
    df=df.sort_index()
    df.fillna(0, inplace=True)
    df['AllStim']=df['OneStim']+df['TwoStim']

    # collapse over block (before vs. after experiment)
    insert2 = df.groupby(level=(0,2)).transform('sum').drop(index=2, level=1).rename(index={1: 3})
    df = pd.concat([df, insert2], sort=False)
    df=df.sort_index()

    # calculate correct response percentages + save table
    for sub in subjects:
        for block in [1, 2, 3]:
            perc1 = (df.loc[(sub,block,'correct'),'OneStim'] / np.nansum(df.loc[(sub, block),'OneStim']))
            perc2 = df.loc[(sub, block, 'correct'), 'TwoStim'] / np.nansum(df.loc[(sub, block ),'TwoStim'])
            percAll = df.loc[(sub, block, 'correct'), 'AllStim'] / np.nansum(df.loc[(sub, block,),'AllStim'])
            insert = pd.DataFrame(index=[(sub, block, 'perc')],
                                  data={'OneStim': perc1, 'TwoStim': perc2, 'AllStim': percAll})
            df = pd.concat([df, insert], sort=False)
    df=df.sort_index()
    if save_df:
        df.to_csv(result_path + 'DetectionTask.csv')

elif load_type == 'df':
    df = pd.read_csv(result_path + 'DetectionTask.csv', index_col=[0,1,2])

else:
    raise Exception('No data was loaded, please enter either "events" or "df" as load_type!')

# crop dataset, convert values to percentages, convert to long format
df = df.drop(columns="AllStim")
df = df.xs('perc',level='decision_category')
df = df.xs(3, level='Run')
df = df*100
df = pd.melt(df,value_vars=['OneStim','TwoStim'],var_name='N_Stim',ignore_index=False).reset_index()
df.rename(columns={'value':'% correct'}, inplace=True)

# boxplot (Fig. 2a)
fig, ax = plt.subplots(figsize = (1,4/3))
ax = sns.boxplot(x='N_Stim',y='% correct',data=df, palette={'OneStim':'.8','TwoStim':'.4'}, showcaps=False)
ax.set_ylim([45,105])
ax.set_ylabel('Correct responses [%]')
ax.set_yticks(np.arange(50,101,10))
ax.set_xlabel('Number of stimuli')
ax.set_xticklabels(['One Stimulus', 'Two Stimuli'])
ax.set_title('Performance control task')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
if save_type:
    plt.savefig(figure_path + 'discrimination_boxplot_includedSubjects.' + save_type)
plt.show()