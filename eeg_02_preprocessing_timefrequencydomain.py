#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
preprocessing of eeg data - time-frequency domain (maximal processing)
----------------------------------------------------------------------
header script applying functions in prepro_functions.py

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

import prepro_functions as pp
import os

subjects = ['es01', 'es03', 'es04', 'es06', 'es07', 'es08', 'es09', 'es10', 'es11', 'es12', 'es13', 'es15', 'es17',
            'es18', 'es19', 'es20', 'es21', 'es22', 'es23', 'es25', 'es26', 'es27', 'es28', 'es30', 'es31', 'es32',
            'es33', 'es34', 'es35', 'es40', 'es41', 'es42', 'es44', 'es45', 'es46', 'es47']

data_path = '/data/pt_02299/ExpSupp/Main/raw_data/'
save_path = '/data/pt_02299/ExpSupp/Main/derivatives/'
montage_file = '/data/pt_02299/ExpSupp/Main/code/standard-10-5-cap.elp'

run_1_ReadFilter = 0
run_2_ManualRejectionPreICA = 0
run_3_ICA = 0
run_4_prepForMuscleTool = 0
# move to MATLAB (eeg_03_preprocessing_muscleArtifactRemoval.m) here
run_5_checkMuscleTool = 0
run_6_ICAComponentSelection = 0
run_7_ICACheck = 0
run_8_AutoRejection = 0
run_9_prepForTFAbasedRejection = 0
# move to MATLAB (eeg_04_preprocessing_tfrSingletrialPlot.m) here
# decide which epochs to drop (create text file tfr/sub-esxx_TFRbasedManualRejection.txt)
run_10_ManualRejectionPostICA = 0
run_11_prepareLEP = 0
run_12_prepareTFA = 0

sr = 500                             # sampling rate for downsampling before prepro
l_freq = 1                           # high-pass edge frequency for ICA
muscle_correct_type = 'corrected'    # whether muscle tool was used or not
threshold_amp = 100                  # epochs containing values exceeding +- this value µV after ICA cleaning will be dropped automatically
threshold_jump = 50                  # epochs containing jumps between adjacent datapoints exceeding +- this value µV after ICA cleaning will be dropped automatically
h_freq = 30                          # low-pass edge frequency for LEPs

for sub in subjects:
    if not os.path.isdir(os.path.join(save_path, 'sub-{}'.format(sub), 'eeg')):
        os.makedirs(os.path.join(save_path, 'sub-{}'.format(sub), 'eeg'))

    # determine correct number of runs per participant
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

    if run_1_ReadFilter:
        pp.readFilter(subject=sub, data_path=data_path, l_freq=l_freq, sr_new=sr, save_path=save_path, n_runs=n_runs)
        
    if run_2_ManualRejectionPreICA:
        pp.manRejectPre(subject=sub, data_path=save_path)

    if run_3_ICA:
        pp.runICA(subject=sub, data_path=save_path, montage_file=montage_file)

    if run_4_prepForMuscleTool:
        pp.prepareForMuscleTool(subject=sub, data_path=save_path)

    if run_5_checkMuscleTool:
        pp.checkMuscleTool(subject=sub, data_path=save_path)

    if run_6_ICAComponentSelection:
        pp.ICSelection(subject=sub, data_path=save_path, montage_file=montage_file, epoch_type=muscle_correct_type)

    if run_7_ICACheck:
        pp.ICACheck(subject=sub, data_path=save_path, epoch_type=muscle_correct_type)

    if run_8_AutoRejection:
        pp.autoRejectPostICA(subject=sub, data_path=save_path, threshold_amp=threshold_amp, threshold_jump=threshold_jump, epoch_type=muscle_correct_type)

    if run_9_prepForTFAbasedRejection:
        pp.prepareForTFARejection(subject=sub, data_path=save_path, montage_file=montage_file, epoch_type=muscle_correct_type, threshold_jump=threshold_jump, sr=sr)

    if run_10_ManualRejectionPostICA:
        pp.manRejectPost(subject=sub, data_path=save_path, epoch_type=muscle_correct_type, threshold_jump=threshold_jump)

    if run_11_prepareLEP:
        pp.prepareLEP(subject=sub, data_path=save_path, threshold_jump=threshold_jump, h_freq=h_freq, montage_file=montage_file, epoch_type=muscle_correct_type)

    if run_12_prepareTFA:
        pp.prepareTFA(subject=sub, data_path=save_path, montage_file=montage_file, epoch_type=muscle_correct_type, threshold_jump=threshold_jump, sr=sr)
