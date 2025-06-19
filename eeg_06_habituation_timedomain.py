#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
habituation of eeg data - analysis in time-domain
-------------------------------------------------
loads data and crops to first stimulus
splitting up trials in two halfes
determine group level latency of N1, N2, P2
visual inspection
determine participant-wise latencies and amplitudes of N1, N2, P2
visual inspection
export table for later analysis in JASP

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
import os, mne
import matplotlib.pyplot as plt

subjects = ['es01', 'es03', 'es04', 'es06', 'es07', 'es08', 'es09', 'es10', 'es11', 'es12', 'es13', 'es15', 'es17',
            'es18', 'es19', 'es20', 'es21', 'es22', 'es23', 'es25', 'es26', 'es27', 'es28', 'es30', 'es31', 'es32',
            'es33', 'es34', 'es35', 'es40', 'es41', 'es42', 'es44', 'es45', 'es46', 'es47']

save_path = '/data/pt_02299/ExpSupp/Main/derivatives/'
result_path = '/data/pt_02299/ExpSupp/Main/results/eeg/'
montage_file = '/data/pt_02299/ExpSupp/Main/code/standard-10-5-cap.elp'

av_14 = []
av_58 = []
fz_14 = []
fz_58 = []

for sub in subjects:
    # load data and events
    events_tsv = pd.read_csv(os.path.join(save_path, 'sub-{}'.format(sub), 'eeg',
                                          'sub-{0}_minimalProcessing_events.tsv'.format(sub)), sep='\t')
    epochs_av = mne.read_epochs(os.path.join(save_path, 'sub-{}'.format(sub), 'eeg',
                                             'sub-{0}_minimalProcessing_ref-average_epo.fif'.format(sub)),
                                preload=True)
    epochs_fz = mne.read_epochs(os.path.join(save_path, 'sub-{}'.format(sub), 'eeg',
                                             'sub-{0}_minimalProcessing_ref-Fz_epo.fif'.format(sub)), preload=True)

    events_tsv = events_tsv[events_tsv.dropped == 0]
    events_tsv.reset_index()

    # crop to include first stimulus only
    epochs_av.crop(-0.5, 1)
    epochs_fz.crop(-0.5, 1)

    # get channel positions
    montage = mne.channels.read_custom_montage(montage_file)
    epochs_av.set_montage(montage)
    epochs_fz.set_montage(montage)

    # split first and second half of experiment
    epochs_av_14 = epochs_av[events_tsv.Run <= 4]
    epochs_av_58 = epochs_av[events_tsv.Run >= 5]
    epochs_fz_14 = epochs_fz[events_tsv.Run <= 4]
    epochs_fz_58 = epochs_fz[events_tsv.Run >= 5]

    # add average per participant
    av_14.append(epochs_av_14.average())
    av_58.append(epochs_av_58.average())
    fz_14.append(epochs_fz_14.average())
    fz_58.append(epochs_fz_58.average())

# calculate grand average
ga_av_14 = mne.grand_average(av_14)
ga_av_58 = mne.grand_average(av_58)
ga_fz_14 = mne.grand_average(fz_14)
ga_fz_58 = mne.grand_average(fz_58)

# plot difference between blocks 1-4 and 5-8
mne.viz.plot_compare_evokeds({'Blocks 1-4':av_14, 'Blocks 5-8':av_58},
                             picks=['Cz'], title='N2P2, Cz-average',
                             colors={'Blocks 1-4':'chocolate','Blocks 5-8':'sandybrown'})
plt.show()
mne.viz.plot_compare_evokeds({'Blocks 1-4':fz_14, 'Blocks 5-8':fz_58},
                             picks=['T8'], title='N1, T8-Fz',
                             colors={'Blocks 1-4':'chocolate','Blocks 5-8':'sandybrown'})
plt.show()

# get and check latencies of N1, N2, P2
_, N2_lat_avg_14 = ga_av_14.copy().pick(['Cz']).get_peak(mode='neg', tmin=0, tmax=1)
_, P2_lat_avg_14 = ga_av_14.copy().pick(['Cz']).get_peak(mode='pos', tmin=0, tmax=1)
_, N1_lat_avg_14 = ga_fz_14.copy().pick(['T8']).get_peak(mode='neg', tmin=0, tmax=N2_lat_avg_14)
mne.viz.plot_compare_evokeds(ga_av_14.copy().crop(0,1),picks=['Cz'],vlines = [N2_lat_avg_14,P2_lat_avg_14])
mne.viz.plot_compare_evokeds(ga_fz_14,picks=['T8'],vlines = [N1_lat_avg_14])

_, N2_lat_avg_58 = ga_av_58.copy().pick(['Cz']).get_peak(mode='neg', tmin=0, tmax=1)
_, P2_lat_avg_58 = ga_av_58.copy().pick(['Cz']).get_peak(mode='pos', tmin=0, tmax=1)
_, N1_lat_avg_58 = ga_fz_58.copy().pick(['T8']).get_peak(mode='neg', tmin=0, tmax=N2_lat_avg_58)
mne.viz.plot_compare_evokeds(ga_av_58.copy().crop(0,1),picks=['Cz'],vlines = [N2_lat_avg_58,P2_lat_avg_58])
mne.viz.plot_compare_evokeds(ga_fz_58,picks=['T8'],vlines = [N1_lat_avg_58])


sub_list = []
N1_14_lat = []
N2_14_lat = []
P2_14_lat = []
N1_58_lat = []
N2_58_lat = []
P2_58_lat = []
N1_14 = []
N1_58 = []
N2_14 = []
N2_58 = []
P2_14 = []
P2_58 = []

for i_sub, sub in enumerate(subjects):
    # get subject-specific latencies and visually ckeck them
    _, N2_lat_sub_14 = av_14[i_sub].copy().pick(['Cz']).get_peak(mode='neg', tmin=(N2_lat_avg_14 - 0.1),
                                                           tmax=(N2_lat_avg_14 + 0.1))
    _, P2_lat_sub_14 = av_14[i_sub].copy().pick(['Cz']).get_peak(mode='pos', tmin=(P2_lat_avg_14 - 0.1),
                                                           tmax=(P2_lat_avg_14 + 0.1))
    N1_14_data = fz_14[i_sub].copy().pick(['T8']).crop(N2_lat_sub_14-0.075, N2_lat_sub_14).to_data_frame()
    idx_N1_lat_sub_14 = N1_14_data.idxmin()["T8"]
    N1_lat_sub_14 = N1_14_data.iloc[idx_N1_lat_sub_14,0]  #0 is index of time column
    mne.viz.plot_compare_evokeds(av_14[i_sub].copy().crop(0, 1), picks=['Cz'], vlines=[N2_lat_sub_14, P2_lat_sub_14],
                                 title=sub)
    mne.viz.plot_compare_evokeds(fz_14[i_sub].copy().crop(0,1),picks=['T8'],vlines = [N1_lat_sub_14], title=sub)
    plt.show(block = True)

    _, N2_lat_sub_58 = av_58[i_sub].copy().pick(['Cz']).get_peak(mode='neg', tmin=(N2_lat_avg_58 - 0.1),
                                                           tmax=(N2_lat_avg_58 + 0.1))
    _, P2_lat_sub_58 = av_58[i_sub].copy().pick(['Cz']).get_peak(mode='pos', tmin=(P2_lat_avg_58 - 0.1),
                                                           tmax=(P2_lat_avg_58 + 0.1))
    N1_58_data = fz_58[i_sub].copy().pick(['T8']).crop(N2_lat_sub_58 - 0.075, N2_lat_sub_58).to_data_frame()
    idx_N1_lat_sub_58 = N1_58_data.idxmin()["T8"]
    N1_lat_sub_58 = N1_58_data.iloc[idx_N1_lat_sub_58, 0]  # 0 is index of time column
    mne.viz.plot_compare_evokeds(av_58[i_sub].copy().crop(0, 1), picks=['Cz'], vlines=[N2_lat_sub_58, P2_lat_sub_58],
                                 title=sub)
    mne.viz.plot_compare_evokeds(fz_58[i_sub].copy().crop(0,1),picks=['T8'],vlines = [N1_lat_sub_58], title=sub)
    plt.show(block = True)

    sub_list.append(sub)
    N1_14_lat.append(N1_lat_sub_14)
    N2_14_lat.append(N2_lat_sub_14)
    P2_14_lat.append(P2_lat_sub_14)
    N1_58_lat.append(N1_lat_sub_58)
    N2_58_lat.append(N2_lat_sub_58)
    P2_58_lat.append(P2_lat_sub_58)

    # get subject- and block-specific peak amplitudes
    N1_14.append(fz_14[i_sub].copy().pick(['T8']).crop(N1_lat_sub_14 - 0.015, N1_lat_sub_14 + 0.015).data.mean() * 1e6)
    N2_14.append(av_14[i_sub].copy().pick(['Cz']).crop(N2_lat_sub_14 - 0.015, N2_lat_sub_14 + 0.015).data.mean() * 1e6)
    P2_14.append(av_14[i_sub].copy().pick(['Cz']).crop(P2_lat_sub_14 - 0.015, P2_lat_sub_14 + 0.015).data.mean() * 1e6)
    N1_58.append(fz_58[i_sub].copy().pick(['T8']).crop(N1_lat_sub_58 - 0.015, N1_lat_sub_58 + 0.015).data.mean() * 1e6)
    N2_58.append(av_58[i_sub].copy().pick(['Cz']).crop(N2_lat_sub_58 - 0.015, N2_lat_sub_58 + 0.015).data.mean() * 1e6)
    P2_58.append(av_58[i_sub].copy().pick(['Cz']).crop(P2_lat_sub_58 - 0.015, P2_lat_sub_58 + 0.015).data.mean() * 1e6)

#save for later analysis in JASP
amplitudes = pd.DataFrame(
    {'subject': sub_list, 'N1_latency_blocks14': N1_14_lat, 'N2_latency_blocks14': N2_14_lat,
     'P2_latency_blocks14': P2_14_lat, 'N1_latency_blocks58': N1_58_lat, 'N2_latency_blocks58': N2_58_lat,
     'P2_latency_blocks58': P2_58_lat, 'N1_blocks14': N1_14, 'N2_blocks14': N2_14, 'P2_blocks14': P2_14,
     'N1_blocks58': N1_58, 'N2_blocks58': N2_58, 'P2_blocks58': P2_58})
amplitudes["N2P2_blocks14"] = amplitudes["P2_blocks14"] - amplitudes["N2_blocks14"]
amplitudes["N2P2_blocks58"] = amplitudes["P2_blocks58"] - amplitudes["N2_blocks58"]
amplitudes.to_csv(os.path.join(result_path, 'amplitude_habituation_N2P2.csv'),
                  index=False, na_rep='n/a')