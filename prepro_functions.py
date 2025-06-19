#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
functions for preprocessing of EEG data

author: Lisa-Marie Pohle

date: 5th June 2025
"""

import mne # version > 1.1, we used 1.2.2
import pandas as pd # we used version 1.4.1
from mne_bids import BIDSPath, read_raw_bids, write_raw_bids # we used version 0.12
# also requires pybv version >= 0.7.3, we used 0.7.5
import os
import numpy as np # we used version 1.23.5
from scipy.io import savemat, loadmat # we used version 1.8.0
import matplotlib.pyplot as plt # we used version 3.6.2
import matplotlib
matplotlib.use('TkAgg')

# reads data, resamples to sr_new and filters with 4th order Butterworth high-pass filter with l_freq and a Notch filter,
# saves filtered data as epochs from -2 to 4s around 1st stimuli, removes bad trials (laser aborted etc.)
def readFilter(subject, data_path, l_freq, sr_new, save_path, n_runs=8, task='expsupp'):
    raw = []
    event_files = []
    for i in range(n_runs):
        run = i + 1   #get correct run number
        bids_path = BIDSPath(subject=subject, run=run, task=task, root=data_path)
        data = read_raw_bids(bids_path=bids_path, extra_params={'preload': True})
        raw.append(data)

        event_file = pd.read_csv(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg', 'sub-{0}_task-{1}_run-0{2}_events.tsv'.format(subject, task, run)), sep='\t')
        event_files.append(event_file)

    mne.concatenate_raws(raw)
    raw = raw[0]

    raw.resample(sr_new)

    raw.filter(l_freq = l_freq, h_freq=None, method='iir')  #per default 4th order butterworth filter_type
    for freq in np.arange(50, 250, 50):
        raw.notch_filter(freqs=freq, method='iir')

    events, ids = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events, ids, tmin=-2, tmax=4, baseline=None)

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
        if ((len(ann) > 1) and ('Abort' in ann[1][2]) and (ann[1][0] < 0.1)): # <0.1 since second annotation should have a timestamp of ~0
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
    events_temp = events_tsv.drop(events_tsv[events_tsv.dropped==1].index)
    events_temp.reset_index(drop=False, inplace=True)
    abort_idx=np.where(events_temp.aborted!=0)[0]
    epochs_subset.drop(abort_idx, "LASER ABORT")
    old_idx = events_temp.loc[abort_idx, 'index'].tolist()
    events_tsv.loc[old_idx, 'dropped'] = 1
    events_tsv.loc[old_idx, 'drop reason'] = 'LASER ABORT'

    events_tsv.to_csv(os.path.join(save_path, 'sub-{}'.format(subject), 'eeg', 'sub-{}_woAborts_hpfilter_events.tsv'.format(subject)),
                      sep='\t', index=False, na_rep='n/a')
    epochs_subset.save(os.path.join(save_path, 'sub-{}'.format(subject), 'eeg', 'sub-{}_woAborts_hpfilter_epo.fif'.format(subject)))

# plots single epochs, user can manually reject an epoch by clicking on it or reject a channel by clicking on the name
def manRejectPre(subject, data_path):
    epochs = mne.read_epochs(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg', 'sub-{}_woAborts_hpfilter_epo.fif'.format(subject)), preload=True)
    events_tsv = pd.read_csv(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg', 'sub-{}_woAborts_hpfilter_events.tsv'.format(subject)), sep='\t')

    if not len(events_tsv) == len(epochs.drop_log):
        raise Exception("Lengths of events file and drop log don't match!")

    epochs.plot(n_epochs=1, n_channels=32, block=True)
    drop_idx = [x for x, y in enumerate(epochs.drop_log) if y == ('USER',)]
    events_tsv.loc[drop_idx, 'dropped'] = 1
    events_tsv.loc[drop_idx, 'drop reason'] = 'MAN REJECT PRE ICA'

    if not len(events_tsv) == len(epochs.drop_log):
        raise Exception("Lengths of events file and drop log don't match!")

    events_tsv.to_csv(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg', 'sub-{}_woAborts_hpfilter_manReject_events.tsv'.format(subject)),
                      sep='\t', index=False, na_rep='n/a')
    epochs.save(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg', 'sub-{}_woAborts_hpfilter_manReject_epo.fif'.format(subject)))

# run the ICA with extended infomax algorithm using the montage file
def runICA(subject, data_path, montage_file):
    epochs = mne.read_epochs(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg', 'sub-{}_woAborts_hpfilter_manReject_epo.fif'.format(subject)),
                             preload=True)
    montage = mne.channels.read_custom_montage(montage_file)
    epochs.set_montage(montage)
    ica = mne.preprocessing.ICA(method='infomax', fit_params=dict(extended=True), random_state=123)
    ica.fit(epochs)
    ica.save(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg', 'sub-{}_woAborts_hpfilter_manReject_ica.fif'.format(subject)))

# save epoched data in IC-space for further use in Matlab (uses unmixing matrix)
def prepareForMuscleTool(subject, data_path):
    epochs = mne.read_epochs(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                          'sub-{}_woAborts_hpfilter_manReject_epo.fif'.format(subject)),
                             preload=True)
    ica = mne.preprocessing.read_ica(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                                  'sub-{}_woAborts_hpfilter_manReject_ica.fif'.format(subject)))

    sources = ica.get_sources(epochs)
    source_data = sources.get_data()

    savemat(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                         'sub-{}_woAborts_hpfilter_manReject_icspace_epo.mat'.format(subject)),
            {'d_old_all': source_data})

# convert artifact-cleaned data from IC space back to epoched data space
# to be able to compare cleaned and uncleaned epochs
# you got there by using the pca_components and the unmixing matrix
# so to get back use the transposed pca_components and the mixing matrix
def checkMuscleTool(subject, data_path):
    epochs = mne.read_epochs(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                          'sub-{}_woAborts_hpfilter_manReject_epo.fif'.format(subject)),
                             preload=True)
    ica = mne.preprocessing.read_ica(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                                  'sub-{}_woAborts_hpfilter_manReject_ica.fif'.format(subject)))
    data = loadmat(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                'sub-{}_woAborts_hpfilter_manReject_muscleCleaned_icspace_epo.mat'.format(subject)))

    source_data_new = data['variable']

    mixing_step = ica.pca_components_.T @ ica.mixing_matrix_
    data_new = mixing_step @ source_data_new
    data_new += ica.pca_mean_[:, None]
    data_new *= ica.pre_whitener_
    epochs_new = epochs.copy()
    # here one complete channel has been dropped
    if subject == "es44":
        epochs_new._data[:,0:24,:] = data_new[:,0:24,:]
        epochs_new._data[:,25:32,:] = data_new[:,24:,:]
    else:
        epochs_new._data[:, :32, :] = data_new

    evo_new=epochs_new.average()
    evo_new.apply_baseline((None, 0))

    evo_old = epochs.average()
    evo_old.apply_baseline((None, 0))
    for electrode in ['Fp1', 'T7', 'Cz', 'T8', 'O2']:
        mne.viz.plot_compare_evokeds({'raw': evo_old, 'corrected': evo_new}, picks=electrode,
                                     title=subject + ', ' + electrode + '-average')
        plt.show()


    evo_old.set_eeg_reference(['Fz'])
    evo_new.set_eeg_reference(['Fz'])
    for electrode in ['Fp1', 'T7', 'Cz', 'T8', 'O2']:
        mne.viz.plot_compare_evokeds({'raw': evo_old, 'corrected': evo_new}, picks=electrode,
                                     title=subject + ', ' + electrode + '-Fz')
        plt.show()

    epochs_new.save(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                             'sub-{}_woAborts_hpfilter_manReject_muscleCleaned_epo.fif'.format(subject)))

# IC selection, first suggestions for rejection are shown (bad eye and muscle components),
# then the user selects components by typing y/n, in the end all components and their time-courses are shown again
# for double checking
def ICSelection(subject, data_path, montage_file, epoch_type):
    if epoch_type == 'uncorrected':
        epochs = mne.read_epochs(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg', 'sub-{}_woAborts_hpfilter_manReject_epo.fif'.format(subject)),
                             preload=True)
    elif epoch_type == 'corrected':
        epochs = mne.read_epochs(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                              'sub-{}_woAborts_hpfilter_manReject_muscleCleaned_epo.fif'.format(subject)),
                                 preload=True)
    else:
        raise NotImplementedError('Other conditions not implemented')

    if subject == 'es44':
        epochs.drop_channels(['FC5'])

    ica = mne.preprocessing.read_ica(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                                  'sub-{}_woAborts_hpfilter_manReject_ica.fif'.format(subject)))

    montage = mne.channels.read_custom_montage(montage_file)
    epochs.set_montage(montage)

    eog_comp, eog_scores = ica.find_bads_eog(epochs)
    ica.plot_scores(eog_scores, exclude = eog_comp, title='Component scores - EOG Rejection: ' + str(eog_comp))

    muscle_comp, muscle_scores = ica.find_bads_muscle(epochs)
    ica.plot_scores(muscle_scores, exclude=muscle_comp, title='Component scores - Muscle Rejection: ' + str(muscle_comp))

    drop_idx = []
    for comp in range(ica.n_components_):
        ica.plot_properties(epochs, picks=[comp])
        drop = ""
        while not ((drop == 'y') or (drop =='n')):
            drop = input('Do you want to drop this component? (y/n)')
            if drop == 'y':
                drop_idx.append(comp)
            elif drop == 'n':
                pass
            else:
                print('Wrong input')

    ica.exclude = drop_idx
    ica.plot_sources(epochs, block=True)

    if epoch_type == 'uncorrected':
        ica.save(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg', 'sub-{}_woAborts_hpfilter_manReject_ICsRejected_ica.fif'.format(subject)))
    elif epoch_type == 'corrected':
        ica.save(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg', 'sub-{}_woAborts_hpfilter_manReject_muscleCleaned_ICsRejected_ica.fif'.format(subject)))
    else:
        raise NotImplementedError('Other conditions not implemented')

# plots some target electrodes (referenced to nose and Fz) before and after removing the rejected ICA components
def ICACheck(subject, data_path, epoch_type):
    if epoch_type == 'uncorrected':
        epochs = mne.read_epochs(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg', 'sub-{}_woAborts_hpfilter_manReject_epo.fif'.format(subject)),
                             preload=True)
        ica = mne.preprocessing.read_ica(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                                      'sub-{}_woAborts_hpfilter_manReject_ICsRejected_ica.fif'.format(subject)))
    elif epoch_type == 'corrected':
        epochs = mne.read_epochs(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                              'sub-{}_woAborts_hpfilter_manReject_muscleCleaned_epo.fif'.format(subject)),
                                 preload=True)
        ica = mne.preprocessing.read_ica(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                                      'sub-{}_woAborts_hpfilter_manReject_muscleCleaned_ICsRejected_ica.fif'.format(subject)))
    else:
        raise NotImplementedError('Other conditions not implemented')

    evo_pre = epochs.average()
    evo_post = evo_pre.copy()
    evo_pre_Fz = epochs.set_eeg_reference(['Fz']).average()
    evo_post_Fz = evo_pre_Fz.copy()
    ica.apply(evo_post)
    ica.apply(evo_post_Fz)

    fig, ax = plt.subplots(2,5, sharex=True)
    for i, electrode in enumerate(['Fp2', 'T7', 'Cz', 'T8', 'O1']):
        idx = evo_pre.info['ch_names'].index(electrode)
        data_pre = evo_pre.to_data_frame(picks=[idx])
        data_post = evo_post.to_data_frame(picks=[idx])
        # nose referenced ERPs with and without IC rejection
        ax[0,i].plot(data_pre.iloc[:, 1])
        ax[0,i].plot(data_post.iloc[:, 1])
        ax[0,i].set_title(electrode)
        # Fz referenced ERPs with and without IC rejection
        data_pre_Fz = evo_pre_Fz.to_data_frame(picks=[idx])
        data_post_Fz = evo_post_Fz.to_data_frame(picks=[idx])
        ax[1,i].plot(data_pre_Fz.iloc[:, 1])
        ax[1,i].plot(data_post_Fz.iloc[:, 1])
    ax[0,0].set_ylabel("ref: nose")
    ax[1,0].set_ylabel("ref: Fz")
    fig.suptitle(subject)
    fig.legend(['Pre ICA', 'Post ICA'])
    plt.show(block=True)

# automatically rejects epochs containing samples with amplitudes > threshold_amp or jumps between adjacent samples
# which are greater than threshold_jump
def autoRejectPostICA(subject, data_path, threshold_amp, threshold_jump, epoch_type):
    events_tsv = pd.read_csv(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                          'sub-{}_woAborts_hpfilter_manReject_events.tsv'.format(subject)),
                             sep='\t')

    if epoch_type == 'uncorrected':
        epochs = mne.read_epochs(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                              'sub-{}_woAborts_hpfilter_manReject_epo.fif'.format(subject)),
                                 preload=True)
        ica = mne.preprocessing.read_ica(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                                      'sub-{}_woAborts_hpfilter_manReject_ICsRejected_ica.fif'.format(
                                                          subject)))
    elif epoch_type == 'corrected':
        epochs = mne.read_epochs(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                              'sub-{}_woAborts_hpfilter_manReject_muscleCleaned_epo.fif'.format(
                                                  subject)),
                                 preload=True)
        ica = mne.preprocessing.read_ica(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                                      'sub-{}_woAborts_hpfilter_manReject_muscleCleaned_ICsRejected_ica.fif'.format(
                                                          subject)))
    else:
        raise NotImplementedError('Other conditions not implemented')

    ica.apply(epochs)
    df = epochs.to_data_frame()
    df.drop(columns=['time', 'condition', 'VEOG', 'HEOG', 'STI 014'], inplace = True)
    df.drop(columns=epochs.info['bads'], inplace = True)
    drop_jump = []
    drop_amp = []
    for i in set(df.epoch):
        sub_df = df[df.epoch == i]
        sub_df = sub_df.drop(columns='epoch')
        diffs = sub_df.diff()
        if np.where(abs(diffs) > threshold_jump)[0].size>0:
            print('Rejected jumps:')
            print(np.where(abs(diffs) > threshold_jump))
        drop_jump.append(np.any(abs(diffs) > threshold_jump))
        if np.where(abs(sub_df) > threshold_amp)[0].size>0:
            print('Rejected amplitude:')
            print(np.where(abs(sub_df) > threshold_amp))
        drop_amp.append(np.any(abs(sub_df) > threshold_amp))
    drop_idx = np.concatenate((np.where(drop_jump)[0], np.where(drop_amp)[0]))
    drop_idx = np.unique(drop_idx)
    epochs.drop(drop_idx, "AUTO REJECT POST ICA")

    epochs.plot_drop_log(n_max_plot=35, subject=subject)
    print(epochs.info['ch_names'])

    if not len(events_tsv) == len(epochs.drop_log):
        raise Exception("Lengths of events file and drop log don't match!")

    drop_idx = [x for x, y in enumerate(epochs.drop_log) if y == ('AUTO REJECT POST ICA',)]
    events_tsv.loc[drop_idx, 'dropped'] = 1
    events_tsv.loc[drop_idx, 'drop reason'] = 'AUTO REJECT POST ICA'

    if not len(events_tsv) == len(epochs.drop_log):
        raise Exception("Lengths of events file and drop log don't match!")

    if epoch_type == 'uncorrected':
        events_tsv.to_csv(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                       'sub-{0}_woAborts_hpfilter_manReject_ICsRejected_autoReject{1}_events.tsv'.format(
                                           subject, threshold_jump)),
                          sep='\t', index=False, na_rep='n/a')
        epochs.save(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                 'sub-{0}_woAborts_hpfilter_manReject_ICsRejected_autoReject{1}_epo.fif'.format(
                                     subject, threshold_jump)))
    elif epoch_type == 'corrected':
        events_tsv.to_csv(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg', 'sub-{0}_woAborts_hpfilter_manReject_muscleCleaned_ICsRejected_autoReject{1}_events.tsv'.format(subject, threshold_jump)),
                          sep='\t', index=False, na_rep='n/a')
        epochs.save(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                 'sub-{0}_woAborts_hpfilter_manReject_muscleCleaned_ICsRejected_autoReject{1}_epo.fif'.format(
                                     subject, threshold_jump)))
    else:
        raise NotImplementedError('Other conditions not implemented')

# concatenate epochs again to pseudo-continuous format, save it in BrainVision format that can be read by Fieldtrip
# for a first time frequency analysis to detect epochs with remaining muscle artifacts
def prepareForTFARejection(subject, data_path, montage_file, epoch_type, threshold_jump, sr):
    if epoch_type == 'uncorrected':
        epochs = mne.read_epochs(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                              'sub-{0}_woAborts_hpfilter_manReject_ICsRejected_autoReject{1}_epo.fif'.format(
                                     subject, threshold_jump)),
                                 preload=True)
    elif epoch_type == 'corrected':
        epochs = mne.read_epochs(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                              'sub-{0}_woAborts_hpfilter_manReject_muscleCleaned_ICsRejected_autoReject{1}_epo.fif'.format(
                                     subject, threshold_jump)),
                                 preload=True)
    else:
        raise NotImplementedError('Other conditions not implemented')

    epochs.set_eeg_reference('average')
    if len(epochs.info['bads'])>0:
        montage = mne.channels.read_custom_montage(montage_file)
        epochs.set_montage(montage)
        epochs.interpolate_bads()

    epochs_clean = mne.concatenate_epochs([epochs['repetition'],epochs['omission']])
    info = epochs_clean.info
    bids_root = data_path

    data = epochs_clean.get_data()
    data_t = data.transpose(1, 0, 2).reshape((data.shape[1], data.shape[0] * data.shape[2]))
    raw = mne.io.RawArray(data_t, info=info)
    events = mne.find_events(raw)
    # this will again find triggers for first stimulus of each trial and repetition/omission, we only want to pick
    # the first ones to have one event per trial -> exclude events where no second event follows within about 1s
    events = events[np.where(np.diff(events, axis=0)[:, 0] < 1.2*sr)[0]]
    bids_path = BIDSPath(subject=subject, task='prepForTFARejection', run=1, root=bids_root)
    write_raw_bids(raw, bids_path, events=events, event_id={'1st': 1}, allow_preload=True, format='BrainVision')

# drop the epochs that were selected based on the time frequency representation (drop information in text file)
def manRejectPost(subject, data_path, epoch_type, threshold_jump):
    if epoch_type == 'uncorrected':
        events_tsv = pd.read_csv(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                       'sub-{0}_woAborts_hpfilter_manReject_ICsRejected_autoReject{1}_events.tsv'.format(
                                           subject, threshold_jump)), sep='\t')
        epochs = mne.read_epochs(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                 'sub-{0}_woAborts_hpfilter_manReject_ICsRejected_autoReject{1}_epo.fif'.format(
                                     subject, threshold_jump)), preload=True)
    elif epoch_type == 'corrected':
        events_tsv = pd.read_csv(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                              'sub-{0}_woAborts_hpfilter_manReject_muscleCleaned_ICsRejected_autoReject{1}_events.tsv'.format(
                                                  subject, threshold_jump)), sep='\t')
        epochs = mne.read_epochs(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                              'sub-{0}_woAborts_hpfilter_manReject_muscleCleaned_ICsRejected_autoReject{1}_epo.fif'.format(
                                                  subject, threshold_jump)), preload=True)
    else:
        raise NotImplementedError('Other conditions not implemented')

    if not len(events_tsv) == len(epochs.drop_log):
        raise Exception("Lengths of events file and drop log don't match!")

    # prepare transformation from indexes used for selecting bad epochs to MNE indices
    df = events_tsv.copy()
    df = df[['dropped']]
    # the indices in Matlab refer to an array in which all bad epochs were dropped already
    # (+ taking care of python 0 and Matlab 1 start index)
    df = df.drop(np.where(df.dropped==1)[0])
    df = df.reset_index()
    df['index_matlab'] = df.index+1
    df = df.drop(columns = "dropped")
    df['index_drop'] = df.index_matlab-1

    marked = np.loadtxt(os.path.join(data_path, 'sub-{}'.format(subject), 'tfr',
                                     'sub-{0}_TFRbasedManualRejection.txt'.format(subject)),
                        delimiter=',')
    if marked.size == 1:
        marked = [marked.tolist()]
    elif marked.size == 0:
        marked = []

    df = df[df.index_matlab.isin(marked)]

    print(list(df.index_drop))
    epochs.drop(list(df.index_drop), 'MAN REJECT POST ICA')
    # use original index (all events) to save drop reason
    events_tsv.loc[list(df['index']), 'dropped'] = 1
    events_tsv.loc[list(df['index']), 'drop reason'] = 'MAN REJECT POST ICA'

    if not len(events_tsv) == len(epochs.drop_log):
        raise Exception("Lengths of events file and drop log don't match!")

    if epoch_type == 'uncorrected':
        events_tsv.to_csv(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                       'sub-{0}_woAborts_hpfilter_manReject_ICsRejected_autoReject{1}_manReject_events.tsv'.format(
                                           subject, threshold_jump)),
                          sep='\t', index=False, na_rep='n/a')
        epochs.save(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                 'sub-{0}_woAborts_hpfilter_manReject_ICsRejected_autoReject{1}_manReject_epo.fif'.format(
                                     subject, threshold_jump)))
    elif epoch_type == 'corrected':
        events_tsv.to_csv(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                       'sub-{0}_woAborts_hpfilter_manReject_muscleCleaned_ICsRejected_autoReject{1}_manReject_events.tsv'.format(
                                                  subject, threshold_jump)),
                          sep='\t', index=False, na_rep='n/a')
        epochs.save(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                 'sub-{0}_woAborts_hpfilter_manReject_muscleCleaned_ICsRejected_autoReject{1}_manReject_epo.fif'.format(
                                                  subject, threshold_jump)))
    else:
        raise NotImplementedError('Other conditions not implemented')

# for laser-evoked potential analyses the data are low-pass filtered with h_freq, an interval from 0.5 s before
# stimulus onset until stimulus onset is used as baseline, trial counts of different conditions are equalised
# (info about this is added to the events.tsv depending on the experimental blocks to look at) and the data are saved
def prepareLEP(subject, data_path, threshold_jump, h_freq, montage_file, epoch_type):
    if epoch_type == 'uncorrected':
        events_tsv = pd.read_csv(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                              'sub-{0}_woAborts_hpfilter_manReject_ICsRejected_autoReject{1}_manReject_events.tsv'.format(
                                     subject, threshold_jump)), sep='\t')
        epochs = mne.read_epochs(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                              'sub-{0}_woAborts_hpfilter_manReject_ICsRejected_autoReject{1}_manReject_epo.fif'.format(
                                                  subject, threshold_jump)),
                                 preload=True)
    elif epoch_type == 'corrected':
        events_tsv = pd.read_csv(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                              'sub-{0}_woAborts_hpfilter_manReject_muscleCleaned_ICsRejected_autoReject{1}_manReject_events.tsv'.format(
                                                  subject, threshold_jump)), sep='\t')
        epochs = mne.read_epochs(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                              'sub-{0}_woAborts_hpfilter_manReject_muscleCleaned_ICsRejected_autoReject{1}_manReject_epo.fif'.format(
                                                  subject, threshold_jump)),
                                 preload=True)
    else:
        raise NotImplementedError('Other IC rejection methods not yet implemented')

    epochs.set_eeg_reference('average')
    epochs_Fz = epochs.copy().set_eeg_reference(['Fz'])

    epochs.filter(l_freq = None, h_freq=h_freq, method='iir')  #per default 4th order butterworth filter
    epochs_Fz.filter(l_freq = None, h_freq=h_freq, method='iir')

    # interpolating bad channels
    if len(epochs.info['bads'])>0:
        montage = mne.channels.read_custom_montage(montage_file)
        epochs.set_montage(montage)
        epochs.interpolate_bads()
        epochs_Fz.set_montage(montage)
        epochs_Fz.interpolate_bads()

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

    if epoch_type == 'uncorrected':
        events_tsv.to_csv(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                       'sub-{0}_woAborts_hpfilter_manReject_ICsRejected_autoReject{1}_manReject_equalizedCounts_events.tsv'.format(
                                           subject, threshold_jump)),
                          sep='\t', index=False, na_rep='n/a')
        epochs.save(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                 'sub-{0}_woAborts_hpfilter_manReject_ICsRejected_autoReject{1}_manReject_lpfilter_ref-average_epo.fif'.format(
                                     subject, threshold_jump)))
        epochs_Fz.save(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                 'sub-{0}_woAborts_hpfilter_manReject_ICsRejected_autoReject{1}_manReject_lpfilter_ref-Fz_epo.fif'.format(
                                     subject, threshold_jump)))
    elif epoch_type == 'corrected':
        events_tsv.to_csv(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                       'sub-{0}_woAborts_hpfilter_manReject_muscleCleaned_ICsRejected_autoReject{1}_manReject_equalizedCounts_events.tsv'.format(
                                                  subject, threshold_jump)),
                          sep='\t', index=False, na_rep='n/a')
        epochs.save(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                 'sub-{0}_woAborts_hpfilter_manReject_muscleCleaned_ICsRejected_autoReject{1}_manReject_lpfilter_ref-average_epo.fif'.format(
                                     subject, threshold_jump)))
        epochs_Fz.save(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                 'sub-{0}_woAborts_hpfilter_manReject_muscleCleaned_ICsRejected_autoReject{1}_manReject_lpfilter_ref-Fz_epo.fif'.format(
                                     subject, threshold_jump)))
    else:
        raise NotImplementedError('Other conditions not implemented')

# concatenate epochs again to pseudo-continuous format, save it in BrainVision format that can be read by Fieldtrip
# for the final time frequency analysis
def prepareTFA(subject, data_path, montage_file, epoch_type, threshold_jump, sr):
    if epoch_type == 'uncorrected':
        epochs = mne.read_epochs(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                              'sub-{0}_woAborts_hpfilter_manReject_ICsRejected_autoReject{1}_manReject_epo.fif'.format(
                                                  subject, threshold_jump)),
                                 preload=True)
    elif epoch_type == 'corrected':
        epochs = mne.read_epochs(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                              'sub-{0}_woAborts_hpfilter_manReject_muscleCleaned_ICsRejected_autoReject{1}_manReject_epo.fif'.format(
                                                  subject, threshold_jump)),
                                 preload=True)
    else:
        raise NotImplementedError('Other IC rejection methods not yet implemented')

    epochs.set_eeg_reference('average')
    if len(epochs.info['bads'])>0:
        montage = mne.channels.read_custom_montage(montage_file)
        epochs.set_montage(montage)
        epochs.interpolate_bads()

    info = epochs.info
    bids_root = data_path

    data = epochs.get_data()
    data_t = data.transpose(1, 0, 2).reshape((data.shape[1], data.shape[0] * data.shape[2]))
    raw = mne.io.RawArray(data_t, info=info)
    events = mne.find_events(raw)
    # this will again find triggers for first stimulus of each trial and repetition/omission, we only want to pick
    # the first ones to have one event per trial -> exclude events where no second event follows within about 1s
    events = events[np.where(np.diff(events, axis=0)[:, 0] < 1.2 * sr)[0]]
    bids_path = BIDSPath(subject=subject, task='expsupp', root=bids_root)
    write_raw_bids(raw, bids_path, events=events, event_id={'1st': 1}, allow_preload=True, format='BrainVision')
