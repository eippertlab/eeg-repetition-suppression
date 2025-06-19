#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analysis skin conductance data
----------------------------------------
applies band pass filter
downsamples data
create epochs based on laser events
determine outlier subjects / blocks based on missing responses
plot single subject results
plot group results

authors:
--------
Ulrike Horn

contact:
--------
uhorn@cbs.mpg.de

date:
-----
14th June 2024
"""

import numpy as np
import pandas as pd
import os
import glob
import json
from scipy import signal, stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import warnings
import mne.stats
import seaborn as sns

raw_path = '/data/pt_02299/ExpSupp/Main/raw_data/'
output_path = '/data/pt_02299/ExpSupp/Main/derivatives/'
result_path = '/data/pt_02299/ExpSupp/Main/results/scr/'

if not os.path.exists(result_path):
    os.makedirs(result_path)

interval = [-1, 8]  # interval for cutting and plotting the data
response_interval = [1, 6]  # interval for amplitude extraction
new_sr = 100  # Hz that data will be downsampled to
task = 'expsupp'

# all subjects (es36 does not exist)
subjects = np.arange(1, 48)
subjects = np.delete(subjects, 35)

excluded_EEG = [2, 5, 14, 16, 24, 29, 37, 38, 39, 43]

preprocess = False
outlier = False
balance_trials = False
subject_plot = False
group_plot = True
amplitudes = False
habituation = False
amplitudes_plot = False
amplitudes_raincloud_plot = True

first_4_runs = False

np.random.seed(1990)

color_green = (1/255, 129/255, 68/255)
color_blue = (5/255, 112/255, 176/255)
color_greenblue = (3/255, 121/255, 122/255)
color_red = (215/255, 48/255, 31/255)
color_purple = (136/255, 65/255, 157/255)
color_redpurple = (175/255, 56/255, 94/255)
new_rc_params = {"font.family": 'Arial', "font.size": 6, "font.serif": [], "svg.fonttype": 'none'}
mpl.rcParams.update(new_rc_params)

for isub in subjects:
    sub = 'sub-es' + str("{:02d}".format(isub))
    print('subject {}'.format(isub))
    physio_files = glob.glob(raw_path + sub + os.sep + 'eeg' + os.sep + '*' + task + '*brainamp_physio.tsv.gz')
    physio_files = np.array(sorted(physio_files))

    if not os.path.exists(output_path + sub + os.sep + 'scr'):
        os.makedirs(output_path + sub + os.sep + 'scr')

    if preprocess:
        for i, p in enumerate(physio_files):
            df = pd.read_csv(p, header=None, sep='\t')
            name = os.path.splitext(os.path.splitext(p)[0])[0]  # get rid of .tsv.gz
            tmp = name.split('_recording')
            name_stem = tmp[0]
            f = open(name + '.json')
            info = json.loads(f.read())
            df.columns = info['Columns']
            sr = info['SamplingFrequency']
            tmp = name_stem.split('run-')
            run = tmp[1]
            raw = df['skinconductance']

            # open corresponding events file, add block info and reduce to one line per trial
            events = pd.read_table(name_stem + '_events.tsv', sep='\t')
            events['Block'] = int(run)
            events = events.drop(events[events.StimNr == 2].index).reset_index(drop=True)
            print('---------subject {}-------'.format(sub))
            print('all events: {}'.format(len(events)))

            # exclude invalid and aborted trials
            to_delete = events[events['invalid_trials'] == 1].index
            print('invalid events: {}'.format(len(to_delete)))
            events = events.drop(to_delete).reset_index(drop=True)
            to_delete = events[(events['aborted'] == -99) | (events['aborted'] == 1) | (events['aborted'] == 2) | (
                    events['aborted'] == 3)].index
            print('aborted trials: {}'.format(len(to_delete)))
            events = events.drop(to_delete)

            # stack all event tables
            if i == 0:
                events_all = events
            else:
                events_all = pd.concat([events_all, events])
            onsets = np.array(events['onset'] - info['StartTime'])

            # bandpass filter between 0.0159 and 5Hz
            cutoff_freq1 = 0.0159
            cutoff_freq2 = 5
            w1 = cutoff_freq1 / (sr / 2)  # Normalize the frequency
            w2 = cutoff_freq2 / (sr / 2)  # Normalize the frequency
            b, a = signal.butter(1, [w1, w2], 'bandpass')
            filtered = signal.filtfilt(b, a, raw)

            # downsample to new sampling rate
            filtered = signal.decimate(filtered, int(sr / new_sr))

            # build epochs
            if len(onsets) > 0:
                for e, event in enumerate(onsets):
                    # if no data at the end?
                    if event * new_sr > len(filtered):
                        nan_vec = np.repeat(np.nan, (int((event + interval[1]) * new_sr) -
                                                     int((event + interval[0]) * new_sr)))
                        epo = nan_vec
                    else:
                        # get all epochs to zero at the time point of event by subtracting the value there
                        baseline_value = filtered[int(event * new_sr)]
                        if len(filtered) < int((event + interval[1]) * new_sr):
                            nan_vec = np.repeat(np.nan, int((event + interval[1]) * new_sr) - len(filtered))
                            epo = filtered[int((event + interval[0]) * new_sr):
                                           len(filtered)] - baseline_value
                            epo = np.hstack((epo, nan_vec))
                        else:
                            epo = filtered[int((event + interval[0]) * new_sr):
                                           int((event + interval[1]) * new_sr)] - baseline_value

                    # rounding issues sometimes lead to 1 sample missing or too much:
                    if len(epo) == (interval[1]-interval[0])*new_sr - 1:
                        epo = filtered[int((event + interval[0]) * new_sr):
                                       int((event + interval[1]) * new_sr + 1)] - baseline_value
                    elif len(epo) == (interval[1]-interval[0])*new_sr + 1:
                        epo = filtered[int((event + interval[0]) * new_sr):
                                       int((event + interval[1]) * new_sr - 1)] - baseline_value
                    if e == 0:
                        epochs_run = epo
                    else:
                        epochs_run = np.vstack((epochs_run, epo))

                # stack all runs together
                if i == 0:
                    epochs_all = epochs_run
                else:
                    epochs_all = np.vstack((epochs_all, epochs_run))
            else:
                print('one run had no valid trials')

            save_name = output_path + sub + os.sep + 'scr' + os.sep + 'scr_' + task + '_all_epochs'
            np.save(save_name, epochs_all)
            save_name = output_path + sub + os.sep + 'scr' + os.sep + 'scr_' + task + '_all_events.csv'
            events_all.to_csv(save_name, index=False)

            # also save z-scored epochs for each subject
            save_name = output_path + sub + os.sep + 'scr' + os.sep + 'scr_' + task + '_all_epochs_zscored'
            epochs_all_z = stats.zscore(epochs_all, axis=None, nan_policy='omit')
            np.save(save_name, epochs_all_z)

    # find out which subjects have not enough responses
    if outlier:
        save_name = output_path + sub + os.sep + 'scr' + os.sep + 'scr_' + task + '_all_epochs.npy'
        epochs_all = np.load(save_name)
        save_name = output_path + sub + os.sep + 'scr' + os.sep + 'scr_' + task + '_all_events.csv'
        events_all = pd.read_csv(save_name)
        for i, block in enumerate(events_all.Block.unique()):
            epochs_run = epochs_all[events_all['Block'] == block]
            events_run = events_all.loc[events_all['Block'] == block, :].copy()
            # within interval of 1 to 8 s after onset there should be something larger than baseline in more than 50%
            # of the trials of a block
            responses = np.zeros([len(events_run)])
            for j, trial in enumerate(events_run['Trial'].unique()):
                any_response_interval = [1, 8]  # relative to onset 0 which is dependent on epoch cutting interval
                baseline_interval = [-1, 0]
                # find maximum in response interval
                max_ind = np.argmax(epochs_run[j, -interval[0]*new_sr + any_response_interval[0]*new_sr:
                                                  -interval[0]*new_sr + any_response_interval[1]*new_sr])
                # maximum index in absolute interval terms
                abs_max_ind = max_ind - interval[0]*new_sr + any_response_interval[0]*new_sr
                amp = epochs_run[j, abs_max_ind]
                # baseline value
                base = np.mean(epochs_run[j, -interval[0] * new_sr + baseline_interval[0] * new_sr:
                                             -interval[0] * new_sr + baseline_interval[1] * new_sr])
                if amp - base > 0.01:
                    responses[j] = 1
            if sum(responses) < 0.25*len(responses):
                events_run['discard_block'] = 1
            else:
                events_run['discard_block'] = 0
            if i == 0:
                events_all_new = events_run
            else:
                events_all_new = pd.concat([events_all_new, events_run])
        # subjects should have at least one block per condition left
        if np.any(events_all_new['Cond_Block'].str.contains('LOW') & (events_all_new['discard_block'] == 0)) &\
                np.any(events_all_new['Cond_Block'].str.contains('HIGH') & (events_all_new['discard_block'] == 0)):
            events_all_new['discard_subject'] = 0
        else:
            events_all_new['discard_subject'] = 1
            print('Subject ' + sub + ' does not have enough responses and will be discarded')

        save_name = output_path + sub + os.sep + 'scr' + os.sep + 'scr_' + task + '_all_events_outlier.csv'
        events_all_new.to_csv(save_name, index=False)

    if balance_trials:
        # code from MNE epochs.equalize_epoch_counts:
        def minimize_time_diff(t_shorter, t_longer):
            """Find a boolean mask to minimize timing differences."""
            from scipy.interpolate import interp1d
            keep = np.ones((len(t_longer)), dtype=bool)
            # special case: length zero or one
            if len(t_shorter) < 2:  # interp1d won't work
                keep.fill(False)
                if len(t_shorter) == 1:
                    idx = np.argmin(np.abs(t_longer - t_shorter))
                    keep[idx] = True
                return keep
            scores = np.ones((len(t_longer)))
            x1 = np.arange(len(t_shorter))
            # The first set of keep masks to test
            kwargs = dict(copy=False, bounds_error=False, assume_sorted=True)
            shorter_interp = interp1d(x1, t_shorter, fill_value=t_shorter[-1],
                                      **kwargs)
            for ii in range(len(t_longer) - len(t_shorter)):
                scores.fill(np.inf)
                # set up the keep masks to test, eliminating any rows that are already
                # gone
                keep_mask = ~np.eye(len(t_longer), dtype=bool)[keep]
                keep_mask[:, ~keep] = False
                # Check every possible removal to see if it minimizes
                x2 = np.arange(len(t_longer) - ii - 1)
                t_keeps = np.array([t_longer[km] for km in keep_mask])
                longer_interp = interp1d(x2, t_keeps, axis=1,
                                         fill_value=t_keeps[:, -1],
                                         **kwargs)
                d1 = longer_interp(x1) - t_shorter
                d2 = shorter_interp(x2) - t_keeps
                scores[keep] = np.abs(d1, d1).sum(axis=1) + np.abs(d2, d2).sum(axis=1)
                keep[np.argmin(scores)] = False
            return keep

        save_name = output_path + sub + os.sep + 'scr' + os.sep + 'scr_' + task + '_all_events_outlier.csv'
        events_all = pd.read_csv(save_name)
        print('---------subject {}-------'.format(sub))
        if not events_all['discard_subject'].unique()[0]:
            # do it once for all blocks
            ind_unexp_rep = np.where(events_all['trial_type'] == '1st/repetition/unexpected')[0]
            ind_exp_rep = np.where(events_all['trial_type'] == '1st/repetition/expected')[0]
            print('the number of unexpected repetition trials is {}'.format(len(ind_unexp_rep)))
            ind_unexp_omi = np.where(events_all['trial_type'] == '1st/omission/unexpected')[0]
            ind_exp_omi = np.where(events_all['trial_type'] == '1st/omission/expected')[0]
            print('the number of unexpected omission trials is {}'.format(len(ind_unexp_omi)))

            # which one is the shortest
            lst = [ind_unexp_rep, ind_exp_rep, ind_unexp_omi, ind_exp_omi]
            shortest = min(lst, key=len)
            shortest_ind = lst.index(shortest)
            # run function on all indices (for the shortest itself it just selects all)
            selected = np.zeros(len(events_all))
            for i, curr_list in enumerate(lst):
                sel = minimize_time_diff(shortest, curr_list)
                selected[curr_list[sel]] = 1

            selected_df = pd.DataFrame(selected)
            selected_df.to_csv(output_path + sub + os.sep + 'scr' + os.sep + 'scr_' + task +
                               '_selected_events_all_blocks.csv',
                               index=False, header=False)

            # then do it again for the case when you just look into the first 4 runs
            ind_unexp_rep = np.where((events_all['trial_type'] == '1st/repetition/unexpected') &
                                     (events_all['Run'] < 5))[0]
            ind_exp_rep = np.where((events_all['trial_type'] == '1st/repetition/expected') &
                                   (events_all['Run'] < 5))[0]
            print('the number of unexpected repetition trials in the first four runs is {}'.format(len(ind_unexp_rep)))
            ind_unexp_omi = np.where((events_all['trial_type'] == '1st/omission/unexpected') &
                                     (events_all['Run'] < 5))[0]
            ind_exp_omi = np.where((events_all['trial_type'] == '1st/omission/expected') &
                                   (events_all['Run'] < 5))[0]
            print('the number of unexpected omission trials in the first four runs is {}'.format(len(ind_unexp_omi)))

            # which one is the shortest
            lst = [ind_unexp_rep, ind_exp_rep, ind_unexp_omi, ind_exp_omi]
            shortest = min(lst, key=len)
            shortest_ind = lst.index(shortest)
            # run function on all indices (for the shortest itself it just selects all)
            selected = np.zeros(len(events_all))
            for i, curr_list in enumerate(lst):
                sel = minimize_time_diff(shortest, curr_list)
                selected[curr_list[sel]] = 1

            selected_df = pd.DataFrame(selected)
            selected_df.to_csv(output_path + sub + os.sep + 'scr' + os.sep + 'scr_' + task +
                               '_selected_events_4_blocks.csv',
                               index=False, header=False)

    if subject_plot:
        save_name = output_path + sub + os.sep + 'scr' + os.sep + 'scr_' + task + '_all_epochs.npy'
        epochs_all = np.load(save_name)
        #
        # plotting average response over all blocks
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        x = np.linspace(interval[0], interval[1], epochs_all.shape[1])
        mean_hp = np.nanmean(epochs_all, axis=0)
        sem_hp = np.nanstd(epochs_all, axis=0) / np.sqrt(epochs_all.shape[0])
        ax1.plot(x, mean_hp)
        ax1.fill_between(x, mean_hp - sem_hp, mean_hp + sem_hp, alpha=0.2)
        ax1.set(ylabel='skin conductance')
        ax1.set(xlabel='time seconds')
        ax1.axvline(x=0, linestyle="--", c="black")
        ax1.axvline(x=1, linestyle="--", c="black")
        fig1.suptitle('Average skin conductance response', fontsize=14)
        plt.savefig(output_path + sub + os.sep + 'scr' + os.sep + 'scr_' + task + '_mean.png')
        plt.show()

        # read event file for more advanced plots
        save_name = output_path + sub + os.sep + 'scr' + os.sep + 'scr_' + task + '_all_events_outlier.csv'
        events_all = pd.read_csv(save_name)

        # plot repetitions vs omissions
        mean_rep = np.nanmean(epochs_all[events_all['Rep'] == 1], axis=0)
        sem_rep = np.nanstd(epochs_all[events_all['Rep'] == 1], axis=0) / np.sqrt(
            epochs_all[events_all['Rep'] == 1].shape[0])
        mean_omi = np.nanmean(epochs_all[events_all['Rep'] == 0], axis=0)
        sem_omi = np.nanstd(epochs_all[events_all['Rep'] == 1], axis=0) / np.sqrt(
            epochs_all[events_all['Rep'] == 1].shape[0])
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        x = np.linspace(interval[0], interval[1], mean_rep.shape[0])
        ax2.plot(x, mean_rep, label='repetition', color=color_redpurple)
        ax2.fill_between(x, mean_rep - sem_rep, mean_rep + sem_rep, alpha=0.2, color=color_redpurple)
        ax2.plot(x, mean_omi, label='omission', color=color_greenblue)
        ax2.fill_between(x, mean_omi - sem_omi, mean_omi + sem_omi, alpha=0.2, color=color_greenblue)
        ax2.set(ylabel='skin conductance')
        ax2.set(xlabel='time seconds')
        ax2.axvline(x=0, linestyle="--", c="black")
        ax2.axvline(x=1, linestyle="--", c="black")
        fig2.legend(loc='upper right', fontsize=7)
        fig2.suptitle('Average skin conductance response repetitions vs omissions', fontsize=14)
        plt.savefig(output_path + sub + os.sep + 'scr' + os.sep + 'scr_' + task + '_rep_vs_omi.png')
        plt.show()

        if task == 'expsupp':
            # plot unexpected vs expected repetitions
            mean_rep_unexp = np.nanmean(epochs_all[(events_all['Rep'] == 1) &
                                                   (events_all['Cond_Block'].str.contains('LOW'))], axis=0)
            sem_rep_unexp = np.nanstd(epochs_all[(events_all['Rep'] == 1) &
                                                 (events_all['Cond_Block'].str.contains('LOW'))], axis=0) / \
                            np.sqrt(epochs_all[(events_all['Rep'] == 1) &
                                               (events_all['Cond_Block'].str.contains('LOW'))].shape[0])
            mean_rep_exp = np.nanmean(epochs_all[(events_all['Rep'] == 1) &
                                                 (events_all['Cond_Block'].str.contains('HIGH'))], axis=0)
            sem_rep_exp = np.nanstd(epochs_all[(events_all['Rep'] == 1) &
                                                 (events_all['Cond_Block'].str.contains('HIGH'))], axis=0) / \
                            np.sqrt(epochs_all[(events_all['Rep'] == 1) &
                                               (events_all['Cond_Block'].str.contains('HIGH'))].shape[0])
            fig3, ax3 = plt.subplots(figsize=(10, 8))
            x = np.linspace(interval[0], interval[1], mean_rep_unexp.shape[0])
            ax3.plot(x, mean_rep_exp, label='expected repetition', color=color_red)
            ax3.fill_between(x, mean_rep_exp - sem_rep_exp, mean_rep_exp + sem_rep_exp, alpha=0.2, color=color_red)
            ax3.plot(x, mean_rep_unexp, label='unexpected repetition', color=color_purple)
            ax3.fill_between(x, mean_rep_unexp - sem_rep_unexp, mean_rep_unexp + sem_rep_unexp, alpha=0.2, color=color_purple)
            ax3.set(ylabel='skin conductance')
            ax3.set(xlabel='time seconds')
            ax3.axvline(x=0, linestyle="--", c="black")
            ax3.axvline(x=1, linestyle="--", c="black")
            fig3.legend(loc='upper right', fontsize=7)
            fig3.suptitle('Average skin conductance response expected repetitions vs unexpected repetitions', fontsize=14)
            plt.savefig(output_path + sub + os.sep + 'scr' + os.sep + 'scr_' + task + '_exp_rep_vs_unexp_rep.png')
            plt.show()

            # plot unexpected vs expected omissions
            mean_omi_unexp = np.nanmean(epochs_all[(events_all['Rep'] == 0) &
                                                   (events_all['Cond_Block'].str.contains('HIGH'))], axis=0)
            sem_omi_unexp = np.nanstd(epochs_all[(events_all['Rep'] == 0) &
                                                 (events_all['Cond_Block'].str.contains('HIGH'))], axis=0) / \
                            np.sqrt(epochs_all[(events_all['Rep'] == 0) &
                                               (events_all['Cond_Block'].str.contains('HIGH'))].shape[0])
            mean_omi_exp = np.nanmean(epochs_all[(events_all['Rep'] == 0) &
                                                 (events_all['Cond_Block'].str.contains('LOW'))], axis=0)
            sem_omi_exp = np.nanstd(epochs_all[(events_all['Rep'] == 0) &
                                               (events_all['Cond_Block'].str.contains('LOW'))], axis=0) / \
                          np.sqrt(epochs_all[(events_all['Rep'] == 0) &
                                             (events_all['Cond_Block'].str.contains('LOW'))].shape[0])
            fig4, ax4 = plt.subplots(figsize=(10, 8))
            x = np.linspace(interval[0], interval[1], mean_omi_unexp.shape[0])
            ax4.plot(x, mean_omi_exp, label='expected omission', color=color_green)
            ax4.fill_between(x, mean_omi_exp - sem_omi_exp, mean_omi_exp + sem_omi_exp, alpha=0.2, color=color_green)
            ax4.plot(x, mean_omi_unexp, label='unexpected omission', color=color_blue)
            ax4.fill_between(x, mean_omi_unexp - sem_omi_unexp, mean_omi_unexp + sem_omi_unexp, alpha=0.2, color=color_blue)
            ax4.set(ylabel='skin conductance')
            ax4.set(xlabel='time seconds')
            ax4.axvline(x=0, linestyle="--", c="black")
            ax4.axvline(x=1, linestyle="--", c="black")
            fig4.legend(loc='upper right', fontsize=7)
            fig4.suptitle('Average skin conductance response expected omissions vs unexpected omissions',
                          fontsize=14)
            plt.savefig(
                output_path + sub + os.sep + 'scr' + os.sep + 'scr_' + task + '_exp_omi_vs_unexp_omi.png')
            plt.show()

        # plotting single trial responses per block
        n_rows = int(np.ceil(len(physio_files) / 2.0))
        fig5, axes = plt.subplots(n_rows, 2, figsize=(10, 10), sharex=True, sharey=True)
        fig5.suptitle("Overview single trial responses {} discarded {}".format(sub, events_all['discard_subject'].unique()))
        for i, block in enumerate(events_all.Block.unique()):
            # for one (excluded subject) there was just one block so the axes do not have two dimensions,
            # there is only one row
            if n_rows == 1:
                curr_indices = i
            else:
                if i < n_rows:
                    curr_indices = (i, 0)
                else:
                    curr_indices = (i - n_rows, 1)
            epochs_run = epochs_all[events_all['Block'] == block]
            events_run = events_all[events_all['Block'] == block]
            x = np.linspace(interval[0], interval[1], epochs_run.shape[1])
            colormap = cm.get_cmap('viridis')
            num_trials = epochs_run.shape[0]
            c = np.linspace(0, 1, num_trials)
            for t in range(num_trials):
                if i == 0:
                    axes[curr_indices].plot(x, epochs_run[t], label='trial' + str(t + 1), color=colormap(c[t]))
                else:
                    axes[curr_indices].plot(x, epochs_run[t], color=colormap(c[t]))
            axes[curr_indices].set(title="run {} discarded: {}".format(block, events_run['discard_block'].unique()))
            axes[curr_indices].axvline(x=0, linestyle="--", c="black")
            axes[curr_indices].set(ylabel='skin conductance µS')
            axes[curr_indices].set(xlabel='time seconds')
        fig5.legend(loc='upper right', fontsize=7)
        plt.savefig(output_path + sub + os.sep + 'scr' + os.sep + 'scr_' + task + '_single_trial_blocks.png')
        plt.show()

if group_plot:
    zscored = False
    if first_4_runs:
        max_runs = 4
    else:
        max_runs = 8
    final_num_subs = 0
    for s, isub in enumerate(subjects):

        sub = 'sub-es' + str("{:02d}".format(isub))
        print('------- Subject {} ------'.format(isub))
        if isub in excluded_EEG:
            print('Subject {} excluded during EEG analysis'.format(isub))
            continue

        # load epochs
        if zscored:
            save_name = output_path + sub + os.sep + 'scr' + os.sep + 'scr_' + task + '_all_epochs_zscored.npy'
        else:
            save_name = output_path + sub + os.sep + 'scr' + os.sep + 'scr_' + task + '_all_epochs.npy'
        epochs_all = np.load(save_name)

        # load events
        save_name = output_path + sub + os.sep + 'scr' + os.sep + 'scr_' + task + '_all_events_outlier.csv'
        events_all = pd.read_csv(save_name)

        if events_all['discard_subject'].unique()[0]:
            print('Subject {} excluded during SCR analysis'.format(isub))
            continue

        final_num_subs += 1

        # read which of these should be selected when matching the trials
        if first_4_runs:
            selected_df = pd.read_csv(
                output_path + sub + os.sep + 'scr' + os.sep + 'scr_' + task + '_selected_events_4_blocks.csv',
                header=None)
        else:
            selected_df = pd.read_csv(
                output_path + sub + os.sep + 'scr' + os.sep + 'scr_' + task + '_selected_events_all_blocks.csv',
                header=None)
        selected = selected_df.values
        events_all = events_all.drop(np.where(selected == 0)[0]).reset_index(drop=True)
        epochs_all = np.delete(epochs_all, np.where(selected == 0)[0], axis=0)

        print('number of events: {}'.format(len(events_all)))

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                # average conditions of interest within a subject
                mean_rep = np.nanmean(epochs_all[events_all['Rep'] == 1], axis=0)
                mean_omi = np.nanmean(epochs_all[events_all['Rep'] == 0], axis=0)
                mean_rep_unexp = np.nanmean(epochs_all[(events_all['Rep'] == 1) &
                                                       (events_all['Cond_Block'].str.contains('LOW'))], axis=0)
                mean_rep_exp = np.nanmean(epochs_all[(events_all['Rep'] == 1) &
                                                     (events_all['Cond_Block'].str.contains('HIGH'))], axis=0)
                mean_omi_unexp = np.nanmean(epochs_all[(events_all['Rep'] == 0) &
                                                       (events_all['Cond_Block'].str.contains('HIGH'))], axis=0)
                mean_omi_exp = np.nanmean(epochs_all[(events_all['Rep'] == 0) &
                                                     (events_all['Cond_Block'].str.contains('LOW'))], axis=0)
                if s == 0:
                    all_means_rep = mean_rep
                    all_means_omi = mean_omi
                    all_means_rep_exp = mean_rep_exp
                    all_means_rep_unexp = mean_rep_unexp
                    all_means_omi_exp = mean_omi_exp
                    all_means_omi_unexp = mean_omi_unexp
                else:
                    all_means_rep = np.vstack((all_means_rep, mean_rep))
                    all_means_omi = np.vstack((all_means_omi, mean_omi))
                    all_means_rep_exp = np.vstack((all_means_rep_exp, mean_rep_exp))
                    all_means_rep_unexp = np.vstack((all_means_rep_unexp, mean_rep_unexp))
                    all_means_omi_exp = np.vstack((all_means_omi_exp, mean_omi_exp))
                    all_means_omi_unexp = np.vstack((all_means_omi_unexp, mean_omi_unexp))
            except RuntimeWarning:
                print('oh no!')
                print('subject {} has no data in these runs'.format(isub))

    # ------------------------------- repetitions------------------------------------------
    # statistics:
    # non-parametric cluster-level paired t test
    # only do statistics within interval [1 8]
    stat_interval = [1, 8]
    start = int((stat_interval[0] - interval[0]) * new_sr)
    stop = int((stat_interval[1] - interval[0]) * new_sr)
    t_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_1samp_test(all_means_rep_unexp[:, start:stop] -
                                                                               all_means_rep_exp[:, start:stop],
                                                                               threshold=None, n_permutations=10000,
                                                                               seed=1990, out_type='mask', tail=1)
    mean_rep_exp = np.mean(all_means_rep_exp, axis=0)
    sem_rep_exp = np.std(all_means_rep_exp, axis=0) / np.sqrt(all_means_rep_exp.shape[0])
    mean_rep_unexp = np.mean(all_means_rep_unexp, axis=0)
    sem_rep_unexp = np.std(all_means_rep_unexp, axis=0) / np.sqrt(all_means_rep_unexp.shape[0])
    
    fig1, ax1 = plt.subplots(figsize=(2, 4/3))
    x = np.linspace(interval[0], interval[1], all_means_rep_exp.shape[1])
    plt.plot(x, mean_rep_exp, label='expected repetition', color=color_red)
    plt.fill_between(x, mean_rep_exp - sem_rep_exp, mean_rep_exp + sem_rep_exp, alpha=0.2,
                     color=color_red, edgecolor='none')
    plt.plot(x, mean_rep_unexp, label='unexpected repetition', color=color_purple)
    plt.fill_between(x, mean_rep_unexp - sem_rep_unexp, mean_rep_unexp + sem_rep_unexp, alpha=0.2,
                     color=color_purple, edgecolor='none')
    rep_ylim = ax1.get_ylim()
    if zscored:
        new_ylim = (round(rep_ylim[0], 1) - 0.2, rep_ylim[1])
    else:
        new_ylim = (round(rep_ylim[0], 1) - 0.05, rep_ylim[1])
    if type(clusters) is list:
        # get the y position and draw again with these added boxes
        if zscored:
            y_pos = round(rep_ylim[0], 1) - 0.1
            height = 0.05
        else:
            y_pos = round(rep_ylim[0], 1) - 0.025
            height = 0.005
        ax1.set_ylim(new_ylim)
        # only put the label for the first use of the rectangle
        sign_counter = 0
        notsign_counter = 0
        for i_c, c in enumerate(clusters):
            c = c[0]
            if cluster_pv[i_c] < 0.05:
                if sign_counter == 0:
                    rect = plt.Rectangle((x[c.start + start], y_pos),
                                         x[c.stop + start - 1] - x[c.start + start], height, facecolor='gray',
                                         alpha=0.8, label='cluster p-value < 0.05')
                    sign_counter = sign_counter + 1
                else:
                    rect = plt.Rectangle((x[c.start + start], y_pos),
                                         x[c.stop + start - 1] - x[c.start + start], height, facecolor='gray',
                                         alpha=0.8)
                ax1.add_patch(rect)
            else:
                print('I will not display what is not significant')
    if zscored:
        ax1.set(ylabel='Skin conductance (z-scored)')
    else:
        ax1.set(ylabel='Skin conductance (µS)')
    ax1.set(xlabel='Time (s)')
    ax1.axvline(x=0, linestyle=":", c="black", label='stimulus onset', linewidth=1)
    ax1.axvline(x=1, linestyle=":", c="black", linewidth=1)
    ax1.legend(loc='lower right', frameon=False)
    # ax1.set_title('Skin conductance response, Repetition trials')
    sns.despine()
    if zscored:
        to_add = '_zscored'
    else:
        to_add = ''
    plt.savefig(result_path + 'scr_{}_mean_unexp_rep_vs_exp_rep_'.format(task) +
                'until_block{}_N{}{}.svg'.format(max_runs, final_num_subs, to_add))
    plt.show()

    # -----------------------------------omissions------------------------------------------------
    stat_interval = [1, 8]
    start = int((stat_interval[0] - interval[0]) * new_sr)
    stop = int((stat_interval[1] - interval[0]) * new_sr)
    t_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_1samp_test(all_means_omi_unexp[:, start:stop] -
                                                                               all_means_omi_exp[:, start:stop],
                                                                               threshold=None, n_permutations=10000,
                                                                               seed=1990, out_type='mask', tail=1)
    mean_omi_exp = np.mean(all_means_omi_exp, axis=0)
    sem_omi_exp = np.std(all_means_omi_exp, axis=0) / np.sqrt(all_means_omi_exp.shape[0])
    mean_omi_unexp = np.mean(all_means_omi_unexp, axis=0)
    sem_omi_unexp = np.std(all_means_omi_unexp, axis=0) / np.sqrt(all_means_omi_unexp.shape[0])

    fig2, ax2 = plt.subplots(figsize=(2, 4/3))
    x = np.linspace(interval[0], interval[1], all_means_omi_exp.shape[1])
    plt.plot(x, mean_omi_exp, label='expected omission', color=color_green)
    plt.fill_between(x, mean_omi_exp - sem_omi_exp, mean_omi_exp + sem_omi_exp, alpha=0.2,
                     color=color_green, edgecolor='none')
    plt.plot(x, mean_omi_unexp, label='unexpected omission', color=color_blue)
    plt.fill_between(x, mean_omi_unexp - sem_omi_unexp, mean_omi_unexp + sem_omi_unexp, alpha=0.2,
                     color=color_blue, edgecolor='none')
    ax2.set_ylim(new_ylim)
    if type(clusters) is list:
        # get the y position and draw again with these added boxes
        if zscored:
            y_pos = round(rep_ylim[0], 1) - 0.1
            height = 0.05
        else:
            y_pos = round(rep_ylim[0], 1) - 0.025
            height = 0.005
        # only put the label for the first use of the rectangle
        sign_counter = 0
        notsign_counter = 0
        for i_c, c in enumerate(clusters):
            c = c[0]
            if cluster_pv[i_c] < 0.05:
                if sign_counter == 0:
                    rect = plt.Rectangle((x[c.start + start], y_pos),
                                         x[c.stop + start - 1] - x[c.start + start], height, facecolor='gray',
                                         alpha=0.8, label='cluster p-value < 0.05')
                    sign_counter = sign_counter + 1
                else:
                    rect = plt.Rectangle((x[c.start + start], y_pos),
                                         x[c.stop + start - 1] - x[c.start + start], height, facecolor='gray',
                                         alpha=0.8)
                ax2.add_patch(rect)
            else:
                print('I will not display what is not significant')

    if zscored:
        ax2.set(ylabel='Skin conductance (z-scored)')
    else:
        ax2.set(ylabel='Skin conductance (µS)')
    ax2.set(xlabel='Time (s)')
    ax2.axvline(x=0, linestyle=":", c="black", linewidth=1, label='stimulus onset')
    ax2.axvline(x=1, linestyle=":", c="gray", linewidth=1)
    ax2.legend(loc='upper right', frameon=False)
    # ax2.set_title('Skin conductance response, Omission trials')
    sns.despine()
    if zscored:
        to_add = '_zscored'
    else:
        to_add = ''
    plt.savefig(result_path + 'scr_{}_mean_unexp_omi_vs_exp_omi_'.format(task) +
                'until_block{}_N{}{}.svg'.format(max_runs, final_num_subs, to_add))
    plt.show()

    # ------------------------------- repetitions vs omissions ------------------------------------------
    # statistics:
    # non-parametric cluster-level paired t test
    # only do statistics within interval [1 8] where the stimulus is done
    stat_interval = [1, 8]
    start = int((stat_interval[0] - interval[0]) * new_sr)
    stop = int((stat_interval[1] - interval[0]) * new_sr)
    t_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_1samp_test(all_means_rep[:, start:stop] -
                                                                               all_means_omi[:, start:stop],
                                                                               threshold=None, n_permutations=10000,
                                                                               seed=1990, out_type='mask', tail=1)
    mean_rep = np.mean(all_means_rep, axis=0)
    sem_rep = np.std(all_means_rep, axis=0) / np.sqrt(all_means_rep.shape[0])
    mean_omi = np.mean(all_means_omi, axis=0)
    sem_omi = np.std(all_means_omi, axis=0) / np.sqrt(all_means_omi.shape[0])

    fig3, ax3 = plt.subplots(figsize=(2, 4/3))
    x = np.linspace(interval[0], interval[1], all_means_rep.shape[1])
    plt.plot(x, mean_rep, label='repetition', color=color_redpurple)
    plt.fill_between(x, mean_rep - sem_rep, mean_rep + sem_rep, alpha=0.2,
                     color=color_redpurple, edgecolor='none')
    plt.plot(x, mean_omi, label='omission', color=color_greenblue)
    plt.fill_between(x, mean_omi - sem_omi, mean_omi + sem_omi, alpha=0.2,
                     color=color_greenblue, edgecolor='none')
    ax3.set_ylim(new_ylim)
    if type(clusters) is list:
        # get the y position and draw again with these added boxes
        if zscored:
            y_pos = round(rep_ylim[0], 1) - 0.1
            height = 0.05
        else:
            y_pos = round(rep_ylim[0], 1) - 0.025
            height = 0.005
        # only put the label for the first use of the rectangle
        sign_counter = 0
        notsign_counter = 0
        for i_c, c in enumerate(clusters):
            c = c[0]
            if cluster_pv[i_c] < 0.05:
                print('p-value is ' + str(cluster_pv[i_c]))
                print(x[c.start + start])
                print(x[c.stop + start - 1])
                if sign_counter == 0:
                    rect = plt.Rectangle((x[c.start + start], y_pos),
                                         x[c.stop + start - 1] - x[c.start + start], height, facecolor='gray',
                                         alpha=0.8, label='cluster p-value < 0.05')
                    sign_counter = sign_counter + 1
                else:
                    rect = plt.Rectangle((x[c.start + start], y_pos),
                                         x[c.stop + start - 1] - x[c.start + start], height, facecolor='gray',
                                         alpha=0.8)
                ax3.add_patch(rect)
            else:
                print('I will not display what is not significant')
    if zscored:
        ax3.set(ylabel='Skin conductance (z-scored)')
    else:
        ax3.set(ylabel='Skin conductance (µS)')
    ax3.set(xlabel='Time (s)')
    ax3.axvline(x=0, linestyle=":", c="black", linewidth=1, label='stimulus onset')
    ax3.axvline(x=1, linestyle=":", c="black", linewidth=1)
    ax3.legend(loc='upper right', frameon=False)
    # ax3.set_title('Skin conductance response, all trials')
    sns.despine()
    if zscored:
        to_add = '_zscored'
    else:
        to_add = ''
    plt.savefig(result_path + 'scr_{}_mean_rep_vs_omi_'.format(task) +
                'until_block{}_N{}{}.svg'.format(max_runs, final_num_subs, to_add))
    plt.show()

if amplitudes:
    for s, isub in enumerate(subjects):
        sub = 'sub-es' + str("{:02d}".format(isub))
        print('------- Subject {} ------'.format(isub))
        if isub in excluded_EEG:
            print('Subject {} excluded during EEG analysis'.format(isub))
        else:
            # load epochs
            save_name = output_path + sub + os.sep + 'scr' + os.sep + 'scr_' + task + '_all_epochs.npy'
            epochs_all = np.load(save_name)
            save_name = output_path + sub + os.sep + 'scr' + os.sep + 'scr_' + task + '_all_epochs_zscored.npy'
            epochs_all_z = np.load(save_name)
            # load events
            save_name = output_path + sub + os.sep + 'scr' + os.sep + 'scr_' + task + '_all_events_outlier.csv'
            events_all = pd.read_cv(save_name)
            # exclude non-responder
            if events_all['discard_subject'].unique()[0]:
                print('Subject ' + sub + ' was discarded during SCR analysis')
            else:
                events_all.drop(columns=['onset', 'duration', 'sample', 'Baseline_temp', 'ISI', 'ITI_pre', 'Mov_start',
                                         'Pos_Block', 'Stim_dur', 'Task', 'decision', 'decisionRT', 'file', 'Lasershot',
                                         'StimNr', 'trial_type', 'value'], inplace=True)
                events_all['Cond_Block'] = events_all['Cond_Block'].str[:-2]
                events_all['unexpected'] = np.where(((events_all['Rep'] == 1) & (events_all['Cond_Block'] == 'LOW')) |
                                                    ((events_all['Rep'] == 0) & (events_all['Cond_Block'] == 'HIGH')),
                                                    1, 0)

                # read which of the trials should be selected when matching the trials
                if first_4_runs:
                    selected_df = pd.read_csv(
                        output_path + sub + os.sep + 'scr' + os.sep + 'scr_' + task + '_selected_events_4_blocks.csv',
                        header=None)
                else:
                    selected_df = pd.read_csv(
                        output_path + sub + os.sep + 'scr' + os.sep + 'scr_' + task + '_selected_events_all_blocks.csv',
                        header=None)
                selected = selected_df.values
                print('number of events: {}'.format(sum(selected)))

                # collect amplitudes
                amps = []
                amps_z = []
                for j in range(len(events_all)):
                    # find maximum in response interval
                    max_ind = np.argmax(epochs_all[j, -interval[0]*new_sr + response_interval[0]*new_sr:
                                                      -interval[0]*new_sr + response_interval[1]*new_sr])
                    # maximum index in absolute interval terms
                    abs_max_ind = max_ind - interval[0]*new_sr + response_interval[0]*new_sr
                    max_value = epochs_all[j, abs_max_ind]
                    max_value_z = epochs_all_z[j, abs_max_ind]
                    # minimum before that from 0 to max
                    min_value = np.min(epochs_all[j, -interval[0] * new_sr:abs_max_ind])
                    min_value_z = np.min(epochs_all_z[j, -interval[0] * new_sr:abs_max_ind])
                    amp = max_value - min_value
                    amp_z = max_value_z - min_value_z
                    amps.append(amp)
                    amps_z.append(amp_z)
                events_all['amplitude'] = amps
                events_all['amplitude_z-scored'] = amps_z
                events_all['selected'] = selected
                if s == 0:
                    amps_table = events_all
                else:
                    amps_table = pd.concat([amps_table, events_all], ignore_index=True)
    # save the whole table
    amps_table.to_csv(result_path + 'scr_' + task + '_amplitudes_all.csv', index=False, header=True)

    # save versions for which you matched the trials
    amps_table.drop(amps_table[amps_table['selected'] == 0].index, inplace=True)

    # save into table
    if first_4_runs:
        amps_table.to_csv(result_path + 'scr_' + task + '_amplitudes_all_4_runs.csv', index=False, header=True)
    else:
        amps_table.to_csv(result_path + 'scr_' + task + '_amplitudes_all_8_runs.csv', index=False, header=True)

    # build means
    amps_table_means = amps_table.groupby(['Subject', 'Rep', 'Cond_Block'], as_index=False).mean()
    amps_table_means = amps_table_means[['Subject', 'Rep', 'Cond_Block', 'amplitude', 'amplitude_z-scored']]
    if first_4_runs:
        amps_table_means.to_csv(result_path + 'scr_' + task + '_amplitudes_means_4_runs.csv', index=False, header=True)
    else:
        amps_table_means.to_csv(result_path + 'scr_' + task + '_amplitudes_means_8_runs.csv', index=False, header=True)

    # and reformat into wide format for stats
    amps_table_means['Rep'] = amps_table_means['Rep'].astype(str)
    amps_table_wide = amps_table_means.pivot(index='Subject', columns=['Rep', 'Cond_Block'],
                                             values=['amplitude', 'amplitude_z-scored'])
    amps_table_wide.columns = amps_table_wide.columns.map('_'.join)
    amps_table_wide.rename(columns={'amplitude_0_HIGH': 'amp_unexp_omi',
                                    'amplitude_1_HIGH': 'amp_exp_rep',
                                    'amplitude_0_LOW': 'amp_exp_omi',
                                    'amplitude_1_LOW': 'amp_unexp_rep',
                                    'amplitude_z-scored_0_HIGH': 'ampZ_unexp_omi',
                                    'amplitude_z-scored_1_HIGH': 'ampZ_exp_rep',
                                    'amplitude_z-scored_0_LOW': 'ampZ_exp_omi',
                                    'amplitude_z-scored_1_LOW': 'ampZ_unexp_rep'
                                    }, inplace=True)
    amps_table_wide.reset_index(inplace=True)
    # as trial numbers are matched we can just average unexp and exp to get the overall amplitudes for rep and omi
    amps_table_wide['amp_rep'] = amps_table_wide[['amp_exp_rep', 'amp_unexp_rep']].mean(axis=1)
    amps_table_wide['amp_omi'] = amps_table_wide[['amp_exp_omi', 'amp_unexp_omi']].mean(axis=1)
    amps_table_wide['ampZ_rep'] = amps_table_wide[['ampZ_exp_rep', 'ampZ_unexp_rep']].mean(axis=1)
    amps_table_wide['ampZ_omi'] = amps_table_wide[['ampZ_exp_omi', 'ampZ_unexp_omi']].mean(axis=1)

    if first_4_runs:
        amps_table_wide.to_csv(result_path + 'scr_' + task + '_amplitudes_means_wide_4_runs.csv', index=False, header=True)
    else:
        amps_table_wide.to_csv(result_path + 'scr_' + task + '_amplitudes_means_wide_8_runs.csv', index=False, header=True)

if habituation:
    amps_table = pd.read_csv(result_path + 'scr_' + task + '_amplitudes_all.csv')
    amps_table['first_half'] = np.where(amps_table['Run'] < 5, 1, 0)
    amps_table.drop(['Cond_Block', 'RatingExp', 'RatingS1', 'RatingS2', 'Stim_temp', 'Trial', 'X1', 'X2', 'Y1', 'Y2',
                     'aborted', 'invalid_trials', 'unexpected', 'selected'], axis=1, inplace=True)
    avgs = amps_table.groupby(['Subject', 'first_half', 'Rep'], as_index=False).mean()
    avgs['Rep'] = avgs['Rep'].astype(str)
    avgs['first_half'] = avgs['first_half'].astype(str)
    amps_table_wide = avgs.pivot(index='Subject', columns=['Rep', 'first_half'],
                                 values=['amplitude', 'amplitude_z-scored'])
    amps_table_wide.columns = amps_table_wide.columns.map('_'.join)
    amps_table_wide.rename(columns={'amplitude_0_1': 'amp_omi_first',
                                    'amplitude_1_1': 'amp_rep_first',
                                    'amplitude_0_0': 'amp_omi_second',
                                    'amplitude_1_0': 'amp_rep_second',
                                    'amplitude_z-scored_0_1': 'ampZ_omi_first',
                                    'amplitude_z-scored_1_1': 'ampZ_rep_first',
                                    'amplitude_z-scored_0_0': 'ampZ_omi_second',
                                    'amplitude_z-scored_1_0': 'ampZ_rep_second'
                                    }, inplace=True)
    amps_table_wide.reset_index(inplace=True)
    amps_table_wide.to_csv(result_path + 'scr_' + task + '_amplitudes_habituation.csv', index=False, header=True)

if amplitudes_plot:
    if first_4_runs:
        amps_table_wide = pd.read_csv(result_path + 'scr_' + task + '_amplitudes_means_wide_4_runs.csv')
    else:
        amps_table_wide = pd.read_csv(result_path + 'scr_' + task + '_amplitudes_means_wide_8_runs.csv')
    amps = amps_table_wide.drop(labels=['ampZ_unexp_rep', 'ampZ_exp_rep', 'ampZ_unexp_omi', 'ampZ_exp_omi', 'Subject'],
                                axis=1)
    amps_rep = amps.drop(labels=['amp_exp_omi', 'amp_unexp_omi'], axis=1)
    amps_omi = amps.drop(labels=['amp_exp_rep', 'amp_unexp_rep'], axis=1)
    # sns.set_context("talk")
    # sns.set_style("white")
    jitter = 0.05

    # -------- repetitions -------------
    df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=(amps_rep.values.shape[0], 2)),
                               columns=['exp', 'unexp'])
    df_x_jitter += np.arange(2)
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 8), sharey=True)
    fig1.suptitle("Skin conductance amplitudes, repetition trials", fontsize=20)
    b = sns.boxplot(ax=ax1, x=np.repeat(0, len(amps_rep)), y=amps_rep['amp_exp_rep'], color=color_red,
                    width=0.5, native_scale=True, showfliers=False)
    for patch in b.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.7))
    c = sns.boxplot(ax=ax1, x=np.repeat(1, len(amps_rep)), y=amps_rep['amp_unexp_rep'], color=color_purple,
                    width=0.5, native_scale=True, showfliers=False)
    for patch in c.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.7))
    df_exp = pd.DataFrame({'jitter': df_x_jitter['exp'], 'amp': amps_rep['amp_exp_rep']})
    df_unexp = pd.DataFrame({'jitter': df_x_jitter['unexp'], 'amp': amps_rep['amp_unexp_rep']})
    sns.scatterplot(ax=ax1, data=df_exp, x='jitter', y='amp', color=color_red, zorder=100, edgecolor="black")
    sns.scatterplot(ax=ax1, data=df_unexp, x='jitter', y='amp', color=color_purple, zorder=100, edgecolor="black")
    ax1.set_xticks(range(2))
    ax1.set_xticklabels(['expected', 'unexpected'])
    ax1.tick_params(axis='x', which='major', labelsize=15)
    ax1.set_xlim(-1.0, 2)
    sns.despine()
    ax1.set_xlabel('')
    ax1.set_ylabel(r'Skin conductance ($\mu$S)', fontsize=15)
    plt.subplots_adjust(left=0.15)
    for idx in amps_rep.index:
        ax1.plot(df_x_jitter.loc[idx, ['exp', 'unexp']],
                 amps_rep.loc[idx, ['amp_exp_rep', 'amp_unexp_rep']],
                 color='grey', linewidth=0.5, linestyle='--', zorder=-1)
    if first_4_runs:
        plt.savefig(result_path + 'SCR_amplitudes_within_sub_rep_4_runs.png')
    else:
        plt.savefig(result_path + 'SCR_amplitudes_within_sub_rep.png')
    plt.show()

    # -------- omissions -------------
    df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=(amps_omi.values.shape[0], 2)),
                               columns=['exp', 'unexp'])
    df_x_jitter += np.arange(2)
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 8), sharey=True)
    fig2.suptitle("Skin conductance amplitudes, omission trials", fontsize=20)
    b = sns.boxplot(ax=ax2, x=np.repeat(0, len(amps_omi)), y=amps_omi['amp_exp_omi'], color=color_green,
                    width=0.5, native_scale=True, showfliers=False)
    for patch in b.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.7))
    c = sns.boxplot(ax=ax2, x=np.repeat(1, len(amps_omi)), y=amps_omi['amp_unexp_omi'], color=color_blue,
                    width=0.5, native_scale=True, showfliers=False)
    for patch in c.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.7))
    df_exp = pd.DataFrame({'jitter': df_x_jitter['exp'], 'amp': amps_omi['amp_exp_omi']})
    df_unexp = pd.DataFrame({'jitter': df_x_jitter['unexp'], 'amp': amps_omi['amp_unexp_omi']})
    sns.scatterplot(ax=ax2, data=df_exp, x='jitter', y='amp', color=color_green, zorder=100, edgecolor="black")
    sns.scatterplot(ax=ax2, data=df_unexp, x='jitter', y='amp', color=color_blue, zorder=100, edgecolor="black")
    ax2.set_xticks(range(2))
    ax2.set_xticklabels(['expected', 'unexpected'])
    ax2.tick_params(axis='x', which='major', labelsize=15)
    ax2.set_xlim(-1.0, 2)
    sns.despine()
    ax2.set_xlabel('')
    ax2.set_ylabel(r'Skin conductance ($\mu$S)', fontsize=15)
    plt.subplots_adjust(left=0.15)
    for idx in amps_omi.index:
        ax2.plot(df_x_jitter.loc[idx, ['exp', 'unexp']],
                 amps_omi.loc[idx, ['amp_exp_omi', 'amp_unexp_omi']],
                 color='grey', linewidth=0.5, linestyle='--', zorder=-1)
    if first_4_runs:
        plt.savefig(result_path + 'SCR_amplitudes_within_sub_omi_4_runs.png')
    else:
        plt.savefig(result_path + 'SCR_amplitudes_within_sub_omi.png')
    plt.show()

if amplitudes_raincloud_plot:
    zscored = False
    if first_4_runs:
        amps_table_wide = pd.read_csv(result_path + 'scr_' + task + '_amplitudes_means_wide_4_runs.csv')
    else:
        amps_table_wide = pd.read_csv(result_path + 'scr_' + task + '_amplitudes_means_wide_8_runs.csv')

    if zscored:
        amps = amps_table_wide.drop(labels=['amp_unexp_rep', 'amp_exp_rep', 'amp_unexp_omi', 'amp_exp_omi', 'Subject'],
                                    axis=1)
        amps.rename(columns={"ampZ_exp_rep": "amp_exp_rep", "ampZ_unexp_rep": "amp_unexp_rep",
                             "ampZ_exp_omi": "amp_exp_omi", "ampZ_unexp_omi": "amp_unexp_omi"}, inplace=True)
    else:
        amps = amps_table_wide.drop(labels=['ampZ_unexp_rep', 'ampZ_exp_rep', 'ampZ_unexp_omi', 'ampZ_exp_omi',
                                            'Subject'],
                                    axis=1)
    amps_rep = amps.drop(labels=['amp_exp_omi', 'amp_unexp_omi'], axis=1)
    amps_omi = amps.drop(labels=['amp_exp_rep', 'amp_unexp_rep'], axis=1)

    # sns.set_context("talk")
    # sns.set_style("white")
    jitter = 0.05
    
    figsize = (4/3,4/3)
    
    # -------- repetitions -------------
    df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=(amps_rep.values.shape[0], 2)),
                               columns=['exp', 'unexp'])
    df_x_jitter += [1, 3]
    fig1, ax1 = plt.subplots(1, 1, figsize=figsize, sharey=True)
    # fig1.suptitle("Skin conductance amplitudes, repetition trials")

    v1 = ax1.violinplot(amps_rep['amp_exp_rep'].values, positions=[0.5], showmeans=False, showextrema=False, showmedians=False,
                        side="low", widths=1.2)  # `side` param requires matplotlib 3.9+
    for pc in v1['bodies']:
        pc.set_facecolor(color_red)
        # pc.set_edgecolor('black')
        pc.set_alpha(0.7)

    v2 = ax1.violinplot(amps_rep['amp_unexp_rep'], positions=[3.5], showmeans=False,
                        showextrema=False, showmedians=False,
                        side="high", widths=1.2)
    for pc in v2['bodies']:
        pc.set_facecolor(color_purple)
        # pc.set_edgecolor('black')
        pc.set_alpha(0.7)

    df_exp = pd.DataFrame({'jitter': df_x_jitter['exp'], 'amp': amps_rep['amp_exp_rep']})
    df_unexp = pd.DataFrame({'jitter': df_x_jitter['unexp'], 'amp': amps_rep['amp_unexp_rep']})
    sns.scatterplot(ax=ax1, data=df_exp, x='jitter', y='amp', color=color_red, zorder=100, edgecolor="black", s=15)
    sns.scatterplot(ax=ax1, data=df_unexp, x='jitter', y='amp', color=color_purple, zorder=100, edgecolor="black", s=15)
    ax1.set_xticks([0.5, 3.5])
    ax1.set_xticklabels(['expected', 'unexpected'])
    ax1.tick_params(axis='x', which='major')
    ax1.set_xlim(-1.0, 5)
    sns.despine()
    ax1.set_xlabel('')
    ax1.set_ylabel(r'Skin conductance (µS)')
    plt.subplots_adjust(left=0.15)
    for idx in amps_rep.index:
        ax1.plot(df_x_jitter.loc[idx, ['exp', 'unexp']],
                 amps_rep.loc[idx, ['amp_exp_rep', 'amp_unexp_rep']],
                 color='grey', linewidth=0.5, linestyle='--', zorder=-1)
    if first_4_runs:
        plt.savefig(result_path + 'SCR_amplitudes_raincloud_within_sub_rep_4_runs.svg')
    else:
        plt.savefig(result_path + 'SCR_amplitudes_raincloud_within_sub_rep.svg')
    plt.show()

    # -------- omissions -------------
    df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=(amps_omi.values.shape[0], 2)),
                               columns=['exp', 'unexp'])
    df_x_jitter += [1, 3]
    fig1, ax1 = plt.subplots(1, 1, figsize=figsize, sharey=True)
    # fig1.suptitle("Skin conductance amplitudes, omission trials")

    v1 = ax1.violinplot(amps_omi['amp_exp_omi'].values, positions=[0.5], showmeans=False, showextrema=False,
                        showmedians=False,
                        side="low", widths=1.2)  # `side` param requires matplotlib 3.9+
    for pc in v1['bodies']:
        pc.set_facecolor(color_green)
        # pc.set_edgecolor('black')
        pc.set_alpha(0.7)

    v2 = ax1.violinplot(amps_omi['amp_unexp_omi'], positions=[3.5], showmeans=False,
                        showextrema=False, showmedians=False,
                        side="high", widths=1.2)
    for pc in v2['bodies']:
        pc.set_facecolor(color_blue)
        # pc.set_edgecolor('black')
        pc.set_alpha(0.7)

    df_exp = pd.DataFrame({'jitter': df_x_jitter['exp'], 'amp': amps_omi['amp_exp_omi']})
    df_unexp = pd.DataFrame({'jitter': df_x_jitter['unexp'], 'amp': amps_omi['amp_unexp_omi']})
    sns.scatterplot(ax=ax1, data=df_exp, x='jitter', y='amp', color=color_green, zorder=100, edgecolor="black", s=15)
    sns.scatterplot(ax=ax1, data=df_unexp, x='jitter', y='amp', color=color_blue, zorder=100, edgecolor="black", s=15)
    ax1.set_xticks([0.5, 3.5])
    ax1.set_xticklabels(['expected', 'unexpected'])
    ax1.tick_params(axis='x', which='major')
    ax1.set_xlim(-1.0, 5)
    sns.despine()
    ax1.set_xlabel('')
    ax1.set_ylabel(r'Skin conductance (µS)')
    plt.subplots_adjust(left=0.15)
    for idx in amps_omi.index:
        ax1.plot(df_x_jitter.loc[idx, ['exp', 'unexp']],
                 amps_omi.loc[idx, ['amp_exp_omi', 'amp_unexp_omi']],
                 color='grey', linewidth=0.5, linestyle='--', zorder=-1)
    if first_4_runs:
        plt.savefig(result_path + 'SCR_amplitudes_raincloud_within_sub_omi_4_runs.svg')
    else:
        plt.savefig(result_path + 'SCR_amplitudes_raincloud_within_sub_omi.svg')
    plt.show()

    # -------- repetitions vs omissions -------------
    amps_all = pd.DataFrame({'rep': amps_rep[['amp_exp_rep', 'amp_unexp_rep']].mean(axis=1),
                             'omi': amps_omi[['amp_exp_omi', 'amp_unexp_omi']].mean(axis=1)})
    df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=(amps_all.values.shape[0], 2)),
                               columns=['rep', 'omi'])
    df_x_jitter += [1, 3]
    fig1, ax1 = plt.subplots(1, 1, figsize=figsize, sharey=True)
    # fig1.suptitle("Repetition vs omission")

    v1 = ax1.violinplot(amps_all['rep'].values, positions=[0.5], showmeans=False, showextrema=False,
                        showmedians=False,
                        side="low", widths=1.2)  # `side` param requires matplotlib 3.9+
    for pc in v1['bodies']:
        pc.set_facecolor(color_redpurple)
        # pc.set_edgecolor('black')
        pc.set_alpha(0.7)

    v2 = ax1.violinplot(amps_all['omi'], positions=[3.5], showmeans=False,
                        showextrema=False, showmedians=False,
                        side="high", widths=1.2)
    for pc in v2['bodies']:
        pc.set_facecolor(color_greenblue)
        # pc.set_edgecolor('black')
        pc.set_alpha(0.7)

    df_rep = pd.DataFrame({'jitter': df_x_jitter['rep'], 'amp': amps_all['rep']})
    df_omi = pd.DataFrame({'jitter': df_x_jitter['omi'], 'amp': amps_all['omi']})
    sns.scatterplot(ax=ax1, data=df_rep, x='jitter', y='amp', color=color_redpurple, zorder=100, edgecolor="black", s=15)
    sns.scatterplot(ax=ax1, data=df_omi, x='jitter', y='amp', color=color_greenblue, zorder=100, edgecolor="black", s=15)
    ax1.set_xticks([0.5, 3.5])
    ax1.set_xticklabels(['repetition', 'omission'])
    ax1.tick_params(axis='x', which='major')
    ax1.set_xlim(-1.0, 5)
    sns.despine()
    ax1.set_xlabel('')
    ax1.set_ylabel(r'Skin conductance (µS)')
    plt.subplots_adjust(left=0.15)
    for idx in amps_all.index:
        ax1.plot(df_x_jitter.loc[idx, ['rep', 'omi']],
                 amps_all.loc[idx, ['rep', 'omi']],
                 color='grey', linewidth=0.5, linestyle='--', zorder=-1)
    if first_4_runs:
        plt.savefig(result_path + 'SCR_amplitudes_raincloud_within_sub_rep_omi_4_runs.svg')
    else:
        plt.savefig(result_path + 'SCR_amplitudes_raincloud_within_sub_rep_omi.svg')
    plt.show()

