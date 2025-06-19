#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
preprocessing of eeg data - time domain (minimal processing)
------------------------------------------------------------
concatenate data from all runs
downsamples data
applies band pass filter
create epochs based on laser events
remove problematic epochs/channels
run ICA
select bad ICs
check selection (plot single-subject waveforms and topographies)
rereferencing
baseline correction
determine trials to drop to equalize event counts

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
import mne # version > 1.1, we used 1.2.2
import pandas as pd # we used version 1.4.1
from mne_bids import BIDSPath, read_raw_bids, write_raw_bids # we used version 0.12
import numpy as np # we used version 1.23.5
import matplotlib.pyplot as plt # we used version 3.6.2

subjects = ['es01', 'es02', 'es03', 'es04', 'es05', 'es06', 'es07', 'es08', 'es09', 'es10', 'es11', 'es12', 'es13',
            'es14', 'es15', 'es16', 'es17', 'es18', 'es19', 'es20', 'es21', 'es22', 'es23', 'es24', 'es25', 'es26',
            'es27', 'es28', 'es29', 'es30', 'es31', 'es32', 'es33', 'es34', 'es35', 'es37', 'es38', 'es39', 'es40',
            'es41', 'es42', 'es43', 'es44', 'es45', 'es46', 'es47']

data_path = '/data/pt_02299/ExpSupp/Main/raw_data/'
save_path = '/data/pt_02299/ExpSupp/Main/derivatives/'
montage_file = '/data/pt_02299/ExpSupp/Main/code/standard-10-5-cap.elp'

step1_dataPreparation = True
step2_ICSelection = True
step3_prepareAnalysis = True

sr = 500      # sampling rate for downsampling before prepro
l_freq = 1    # high-pass edge frequency for ICA
h_freq = 30   # low-pass edge frequency for LEPs

for sub in subjects:
    if not os.path.isdir(os.path.join(save_path, 'sub-{}'.format(sub), 'eeg')):
        os.makedirs(os.path.join(save_path, 'sub-{}'.format(sub), 'eeg'))

    if sub == 'es16':
        n_runs = 5
    elif sub == 'es29':
        n_runs = 4
    elif sub == 'es38':
        n_runs = 1
    elif sub == 'es43':
        n_runs = 4
    elif sub == 'es46':
        n_runs = 7
    else:
        n_runs = 8

    if step1_dataPreparation:
        # load raw data
        raw = []
        event_files = []
        for i in range(n_runs):
            run = i + 1  # get correct run number
            bids_path = BIDSPath(subject=sub, run=run, task='expsupp', root=data_path)
            data = read_raw_bids(bids_path=bids_path, extra_params={'preload': True})
            raw.append(data)

            event_file = pd.read_csv(os.path.join(data_path, 'sub-{}'.format(sub), 'eeg',
                                                  'sub-{0}_task-expsupp_run-0{1}_events.tsv'.format(sub, run)),
                                     sep='\t')
            event_files.append(event_file)

        mne.concatenate_raws(raw)
        raw = raw[0]

        # downsampling, filtering, epoching
        raw.resample(sr)
        raw.filter(l_freq=l_freq, h_freq=h_freq)

        events, ids = mne.events_from_annotations(raw)
        epochs = mne.Epochs(raw, events, ids, tmin=-2, tmax=4, baseline=None, preload=True)

        # we want to capture only the epochs around first stimuli (dataset contains 2 triggers per trial - 1st + 2nd
        # stimulus) even if the stimuli are aborts (in case we ever want to calculate everything with these aborted
        # trials included) but using this subset we accidentally capture those that are centered around an abort at
        # the second stimulus of a trial, delete these extra epochs as the ones centered around the first stimulus
        # are already included
        epochs_subset = epochs['1st', 'Abort']
        annotations = epochs_subset.get_annotations_per_epoch()
        drop_idx = []
        # normal annotations example: [(0, 0.125, '1st/omission/expected'), (1, 0.125, '2nd/omission/expected')]
        # abort example to keep: [(0, 0.125, 'Abort')]
        # abort example to keep: [(0, 0.125, 'Abort'), (1, 0.125, 'Abort')]
        # abort examples to delete: [(-1, dur_1, '1st/omission/expected'), (0, dur_2, 'Abort')]
        for a, ann in enumerate(annotations):
            if ((len(ann) > 1) and ('Abort' in ann[1][2]) and (
                    ann[1][0] < 0.1)):  # <0.1 since second annotation should have a timestamp of ~0
                drop_idx.append(a)
        epochs_subset.drop(drop_idx)
        epochs_subset.reset_drop_log_selection()

        events_tsv = pd.concat(event_files)
        events_tsv.reset_index(drop=True, inplace=True)
        events_tsv.drop(events_tsv[events_tsv.StimNr == 2].index, inplace=True)
        events_tsv.reset_index(drop=True, inplace=True)
        events_tsv['dropped'] = 0
        events_tsv['drop reason'] = 'n/a'

        # delete invalid trials (e.g. participant is coughing, etc)
        invalid_idx = np.where(events_tsv.invalid_trials != 0)[0]
        epochs_subset.drop(invalid_idx, "INVALID TRIAL")
        events_tsv.loc[invalid_idx, 'dropped'] = 1
        events_tsv.loc[invalid_idx, 'drop reason'] = 'INVALID TRIAL'

        # delete trials in which laser aborted (but take care of already dropped indices)
        events_temp = events_tsv.drop(events_tsv[events_tsv.dropped == 1].index)
        events_temp.reset_index(drop=False, inplace=True)
        abort_idx = np.where(events_temp.aborted != 0)[0]
        epochs_subset.drop(abort_idx, "LASER ABORT")
        old_idx = events_temp.loc[abort_idx, 'index'].tolist()
        events_tsv.loc[old_idx, 'dropped'] = 1
        events_tsv.loc[old_idx, 'drop reason'] = 'LASER ABORT'

        # for 1 participant channel FC5 was broken, mark it as bad for later interpolation, will be omitted by following analysis
        if sub == 'es44':
            epochs_subset.info["bads"] = ['FC5']

        # prepare data for ICA and run ICA
        montage = mne.channels.read_custom_montage(montage_file)
        epochs_subset.set_montage(montage)
        ica = mne.preprocessing.ICA(method='infomax', fit_params=dict(extended=True), random_state=123)
        ica.fit(epochs_subset)

        # save data
        epochs_subset.save(os.path.join(save_path, 'sub-{}'.format(sub), 'eeg',
                             'sub-{}_minimalProcessing_epo.fif'.format(sub)))
        ica.save(os.path.join(save_path, 'sub-{}'.format(sub), 'eeg',
                                        'sub-{}_minimalProcessing_ica.fif'.format(sub)))
        events_tsv.to_csv(os.path.join(save_path, 'sub-{}'.format(sub), 'eeg',
                                       'sub-{}_minimalProcessing_events.tsv'.format(sub)),
                          sep='\t', index=False, na_rep='n/a')

    if step2_ICSelection:
        # load data
        epochs = mne.read_epochs(os.path.join(save_path, 'sub-{}'.format(sub), 'eeg',
                                              'sub-{}_minimalProcessing_epo.fif'.format(sub)), preload=True)
        ica = mne.preprocessing.read_ica(os.path.join(save_path, 'sub-{}'.format(sub), 'eeg',
                                        'sub-{}_minimalProcessing_ica.fif'.format(sub)))

        if sub == 'es44':
            epochs.drop_channels(['FC5'])

        selection_done = False
        while not selection_done:
            # reject ICs of eyeblinks
            eog_comp, eog_scores = ica.find_bads_eog(epochs)
            ica.plot_scores(eog_scores, exclude=eog_comp, title='Component scores - EOG Rejection: ' + str(eog_comp))

            drop_idx = []
            for comp in range(ica.n_components_):
                ica.plot_properties(epochs, picks=[comp])
                drop = ""
                while not ((drop == 'y') or (drop == 'n')):
                    drop = input('Do you want to drop this component? (y/n)')
                    if drop == 'y':
                        drop_idx.append(comp)
                    elif drop == 'n':
                        pass
                    else:
                        print('Wrong input')

            ica.exclude = drop_idx
            ica.plot_sources(epochs, block=True)

            # check rejection
            evo_pre = epochs.average()
            evo_post = evo_pre.copy()
            evo_pre_Fz = epochs.copy().set_eeg_reference(['Fz']).average()
            evo_post_Fz = evo_pre_Fz.copy()
            ica.apply(evo_post)
            ica.apply(evo_post_Fz)

            fig, ax = plt.subplots(2, 5, sharex=True)
            for i, electrode in enumerate(['Fp2', 'T7', 'Cz', 'T8', 'O1']):
                idx = evo_pre.info['ch_names'].index(electrode)
                data_pre = evo_pre.to_data_frame(picks=[idx])
                data_post = evo_post.to_data_frame(picks=[idx])
                # nose referenced ERPs with and without IC rejection
                ax[0, i].plot(data_pre.iloc[:, 1])
                ax[0, i].plot(data_post.iloc[:, 1])
                ax[0, i].set_title(electrode)
                # Fz referenced ERPs with and without IC rejection
                data_pre_Fz = evo_pre_Fz.to_data_frame(picks=[idx])
                data_post_Fz = evo_post_Fz.to_data_frame(picks=[idx])
                ax[1, i].plot(data_pre_Fz.iloc[:, 1])
                ax[1, i].plot(data_post_Fz.iloc[:, 1])
            ax[0, 0].set_ylabel("ref: nose")
            ax[1, 0].set_ylabel("ref: Fz")
            fig.suptitle(sub)
            fig.legend(['Pre ICA', 'Post ICA'])
            plt.show(block=True)

            fig, ((ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11),
                  (ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20, ax21, ax22)) = plt.subplots(2, 11)
            evo_pre.plot_topomap([0, 0.05, 0.1, 0.15, 0.2, 0.244, 0.264, 0.3, 0.35, 0.4], ch_type="eeg", axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11])
            evo_post.plot_topomap([0, 0.05, 0.1, 0.15, 0.2, 0.244, 0.264, 0.3, 0.35, 0.4], ch_type="eeg", axes = [ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20, ax21, ax22])
            plt.show(block=True)

            #save data
            save = ""
            while not ((save == 'y') or (save == 'n')):
                save = input('Do you want to save this selection? (y/n)')
                if save == 'y':
                    ica.save(os.path.join(save_path, 'sub-{}'.format(sub), 'eeg',
                                          'sub-{}_minimalProcessing_bads_ica.fif'.format(sub)))
                    selection_done = True
                elif save == 'n':
                    pass
                else:
                    print('Wrong input')

    if step3_prepareAnalysis:
        # load data
        epochs = mne.read_epochs(os.path.join(save_path, 'sub-{}'.format(sub), 'eeg',
                                              'sub-{}_minimalProcessing_epo.fif'.format(sub)), preload=True)
        ica = mne.preprocessing.read_ica(os.path.join(save_path, 'sub-{}'.format(sub), 'eeg',
                                          'sub-{}_minimalProcessing_bads_ica.fif'.format(sub)))
        events_tsv = pd.read_csv(os.path.join(save_path, 'sub-{}'.format(sub), 'eeg',
                                       'sub-{}_minimalProcessing_events.tsv'.format(sub)), sep='\t')

        # apply IC selection
        ica.apply(epochs)

        # re-referencing, baseline correction
        epochs.set_eeg_reference('average')
        epochs_Fz = epochs.copy().set_eeg_reference(['Fz'])

        epochs.apply_baseline((-0.5,0))
        epochs_Fz.apply_baseline((-0.5,0))

        event_ids = epochs.event_id
        event_ids.pop('Abort', None)

        # determine which epochs should be dropped to equalise the trial numbers between the conditions for every subset
        # of experimental blocks, save this info to the events.tsv
        for min_block in [1, 3, 5, 7]:
            for max_block in [2, 4, 6, 8]:
                if min_block < max_block:
                    events_temp = events_tsv[events_tsv.dropped == 0]
                    events_temp.reset_index(drop=True, inplace=True)
                    epochs_temp = epochs[(events_temp.Run >= min_block) & (events_temp.Run <= max_block)]
                    try:
                        epochs_temp_short, _ = epochs_temp.copy().equalize_event_counts(event_ids)
                    except:
                        epochs_temp.drop(range(len(epochs_temp)), 'EQUALIZED_COUNT')
                    finally:
                        col_name = 'drop_log_blocks' + str(min_block) + str(max_block)
                        events_tsv[col_name] = epochs_temp_short.drop_log

        events_tsv.to_csv(os.path.join(save_path, 'sub-{}'.format(sub), 'eeg',
                                       'sub-{}_minimalProcessing_final_events.tsv'.format(sub)), sep='\t', index=False,
                          na_rep='n/a')
        epochs.save(os.path.join(save_path, 'sub-{}'.format(sub), 'eeg',
                                 'sub-{}_minimalProcessing_ref-average_epo.fif'.format(sub)), overwrite=True)
        epochs_Fz.save(os.path.join(save_path, 'sub-{}'.format(sub), 'eeg',
                                 'sub-{}_minimalProcessing_ref-Fz_epo.fif'.format(sub)), overwrite=True)