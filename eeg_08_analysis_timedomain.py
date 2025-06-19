#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analysis of eeg data - time-domain
----------------------------------
load data and drop unused trials
calculate average waveform per participant and condition
determine group-level latencies of N1, N2, P2
creates plots for figures 2e, 3a-b, 4a-b, S1a-b, S2a-b, S3a-b, S4a-b
determines individual latencies of N1, N2. P2
calculates participant- and condition-wise amplitudes of N1, N2, P2 for later analysis with JASP
performes cluster-based permutation test

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
import numpy as np
import matplotlib.pyplot as plt
from mne.channels import find_ch_adjacency
from mne.stats import spatio_temporal_cluster_test
import matplotlib as mpl

subjects = ['es01', 'es03', 'es04', 'es06', 'es07', 'es08', 'es09', 'es10', 'es11', 'es12', 'es13', 'es15', 'es17',
            'es18', 'es19', 'es20', 'es21', 'es22', 'es23', 'es25', 'es26', 'es27', 'es28', 'es30', 'es31', 'es32',
            'es33', 'es34', 'es35', 'es40', 'es41', 'es42', 'es44', 'es45', 'es46', 'es47']

save_path = '/data/pt_02299/ExpSupp/Main/derivatives/'
result_path = '/data/pt_02299/ExpSupp/Main/results/eeg/'
figure_path = '/data/pt_02299/ExpSupp/Main/results/figures/'
montage_file = '/data/pt_02299/ExpSupp/Main/code/standard-10-5-cap.elp'

save_type = 'svg'    # enter data format to which figures should be saved (e.g. svg, png), enter None to suppress saving

# what steps to run
create_fig2 = 0       # creates plots for Fig. 2e with setting processing = 'minimal'
create_fig3 = 0       # creates waveform plots for Fig 3a+b with settings processing = 'minimal' and end_block = 4 or Fig S2 with end_block = 8
get_amplitudes = 0    # extracts peak amplitudes per participant and condition for later analysis with JASP
create_fig4 = 0       # creates waveform plots for Fig 4a+b with settings processing = 'minimal' and end_block = 4 or Fig S3 with end_block = 8
create_figS1 = 0      # creates waveform plots for Fig S1a+b with settings processing = 'minimal' and end_block = 4 or Fig S4 with end_block = 8
cbpt = 1              # performes cluster-based permutation tests for repetition and omission trials, run twice with different end_block settings

# data + analysis settings
processing = 'minimal'
start_block = 1
end_block = 8

# plot settings
color_green = (1/255, 129/255, 68/255)
color_blue = (5/255, 112/255, 176/255)
color_greenblue = (3/255, 121/255, 122/255)
color_red = (215/255, 48/255, 31/255)
color_purple = (136/255, 65/255, 157/255)
color_redpurple = (175/255, 56/255, 94/255)
new_rc_params = {"font.family": 'Arial', "font.size": 6, "font.serif": [], "svg.fonttype": 'none'}
mpl.rcParams.update(new_rc_params)

# prepare data structures
fig2_av = []
fig2_fz = []
rep_exp_av = []
rep_une_av = []
omi_exp_av = []
omi_une_av = []
rep_av = []
omi_av = []
av = []
high_av = []
low_av = []
rep_exp_fz = []
rep_une_fz = []
omi_exp_fz = []
omi_une_fz = []
rep_fz = []
omi_fz = []
fz = []
high_fz = []
low_fz = []

# load data and events
for sub in subjects:
    if processing == 'maximal':
        events_tsv = pd.read_csv(os.path.join(save_path, 'sub-{}'.format(sub), 'eeg',
                                              'sub-{0}_woAborts_hpfilter_manReject_muscleCleaned_ICsRejected_autoReject50_equalizedCounts_events.tsv'.format(
                                                  sub)), sep='\t')
        epochs_av = mne.read_epochs(os.path.join(save_path, 'sub-{}'.format(sub), 'eeg',
                                              'sub-{0}_woAborts_hpfilter_manReject_muscleCleaned_ICsRejected_autoReject50_lpfilter_ref-average_epo.fif'.format(
                                                  sub)),
                                 preload=True)
        epochs_fz = mne.read_epochs(os.path.join(save_path, 'sub-{}'.format(sub), 'eeg',
                                              'sub-{0}_woAborts_hpfilter_manReject_muscleCleaned_ICsRejected_autoReject50_lpfilter_ref-Fz_epo.fif'.format(
                                                  sub)),
                                 preload=True)
    elif processing == 'minimal':
        events_tsv = pd.read_csv(os.path.join(save_path, 'sub-{}'.format(sub), 'eeg',
                                              'sub-{0}_minimalProcessing_final_events.tsv'.format(sub)), sep='\t')
        epochs_av = mne.read_epochs(os.path.join(save_path, 'sub-{}'.format(sub), 'eeg',
                                              'sub-{0}_minimalProcessing_ref-average_epo.fif'.format(sub)), preload=True)
        epochs_fz = mne.read_epochs(os.path.join(save_path, 'sub-{}'.format(sub), 'eeg',
                                              'sub-{0}_minimalProcessing_ref-Fz_epo.fif'.format(sub)), preload=True)

    else:
        raise Exception('Please specify processing pipeline correctly')
    montage = mne.channels.read_custom_montage(montage_file)
    epochs_av.set_montage(montage)
    epochs_fz.set_montage(montage)

    # we only want to look at responses to the first stimulus in Fig.2
    epo_av = epochs_av.copy().crop(-0.5,1)
    epo_fz = epochs_fz.copy().crop(-0.5,1)
    fig2_av.append(epo_av.average())
    fig2_fz.append(epo_fz.average())
    # for everything else we want to see responses to full trials
    epochs_av.crop(-1,2)
    epochs_fz.crop(-1,2)

    # drop trials to equalize event counts per condition (for this, first drop all the trials not existend in the eeg
    # data anymore, e.g. due to laser aborts, from the event file and then find indices of all trials which should
    # additionally be dropped, note that this is not done for the data shown in Fig. 2)
    df = events_tsv.copy()
    df = df[['dropped']]
    df['drop_log'] = events_tsv['drop_log_blocks{}{}'.format(start_block, end_block)]
    df = df.drop(np.where(df.dropped == 1)[0])
    df = df.reset_index()
    df = df.drop(columns="dropped")
    df['index_drop'] = df.index
    df = df[(df.drop_log == "('EQUALIZED_COUNT',)")|(df.drop_log=="('IGNORED',)")]
    epochs_av.drop(list(df.index_drop), 'EQUALIZED_COUNT')
    epochs_fz.drop(list(df.index_drop), 'EQUALIZED_COUNT')

    # collect average waveforms per participant and condition
    rep_exp_av.append(epochs_av['1st/repetition/expected'].average())
    rep_une_av.append(epochs_av['1st/repetition/unexpected'].average())
    omi_exp_av.append(epochs_av['1st/omission/expected'].average())
    omi_une_av.append(epochs_av['1st/omission/unexpected'].average())
    rep_av.append(epochs_av['repetition'].average())
    omi_av.append(epochs_av['omission'].average())
    av.append(epochs_av.average())
    high_av.append(epochs_av['1st/repetition/expected','1st/omission/unexpected'].average())
    low_av.append(epochs_av['1st/repetition/unexpected','1st/omission/expected'].average())
    rep_exp_fz.append(epochs_fz['1st/repetition/expected'].average())
    rep_une_fz.append(epochs_fz['1st/repetition/unexpected'].average())
    omi_exp_fz.append(epochs_fz['1st/omission/expected'].average())
    omi_une_fz.append(epochs_fz['1st/omission/unexpected'].average())
    rep_fz.append(epochs_fz['repetition'].average())
    omi_fz.append(epochs_fz['omission'].average())
    fz.append(epochs_fz.average())
    high_fz.append(epochs_fz['1st/repetition/expected','1st/omission/unexpected'].average())
    low_fz.append(epochs_fz['1st/repetition/unexpected','1st/omission/expected'].average())

# calculate grand averages
ga_fig2_av = mne.grand_average(fig2_av)
ga_fig2_fz = mne.grand_average(fig2_fz)
ga_rep_exp_av = mne.grand_average(rep_exp_av)
ga_rep_une_av = mne.grand_average(rep_une_av)
ga_omi_exp_av = mne.grand_average(omi_exp_av)
ga_omi_une_av = mne.grand_average(omi_une_av)
ga_rep_av = mne.grand_average(rep_av)
ga_omi_av = mne.grand_average(omi_av)
ga_av = mne.grand_average(av)
ga_high_av = mne.grand_average(high_av)
ga_low_av = mne.grand_average(low_av)
ga_rep_exp_fz = mne.grand_average(rep_exp_fz)
ga_rep_une_fz = mne.grand_average(rep_une_fz)
ga_omi_exp_fz = mne.grand_average(omi_exp_fz)
ga_omi_une_fz = mne.grand_average(omi_une_fz)
ga_rep_fz = mne.grand_average(rep_fz)
ga_omi_fz = mne.grand_average(omi_fz)
ga_fz = mne.grand_average(fz)
ga_high_fz = mne.grand_average(high_fz)
ga_low_fz = mne.grand_average(low_fz)

# calculate average latencies of N1, N2, P2 based on all trials
_, N2_lat_avg, N2_amp = ga_fig2_av.copy().pick(['Cz']).get_peak(mode='neg', tmin=0, tmax=1, return_amplitude=True)
_, P2_lat_avg, P2_amp = ga_fig2_av.copy().pick(['Cz']).get_peak(mode='pos', tmin=0, tmax=1, return_amplitude=True)
_, N1_lat_avg, N1_amp = ga_fig2_fz.copy().pick(['T8']).get_peak(mode='neg', tmin=0, tmax=N2_lat_avg, return_amplitude=True)
print('AMPLITUDES: N1: ' + str(N1_amp) + ', N2: ' + str(N2_amp) + ', P2: ' + str(P2_amp))
print('LATENCIES: N1: ' + str(N1_lat_avg) + ', N2: ' + str(N2_lat_avg) + ', P2: ' + str(P2_lat_avg))

if create_fig2:
    fig2e_up = mne.viz.plot_evoked_joint(ga_fig2_fz, [N1_lat_avg], ts_args={'highlight': [(0, 0.125)]}, show=False)
    fig2e_up.axes[0].collections[0].set(color='black')   # change color of the area marking the stimulus duration
    fig2e_up.set_size_inches((7.5,4))

    if save_type:
        fig2e_up.savefig(figure_path + 'Fig2/N1_evokedAndTopographies.' + save_type)
    plt.show(block=True)

    fig2e_low = mne.viz.plot_evoked_joint(ga_fig2_av, [N2_lat_avg, P2_lat_avg], ts_args={'highlight': [(0, 0.125)]}, show=False)
    fig2e_low.axes[0].collections[0].set(color='black')   # change color of the area marking the stimulus duration
    fig2e_low.set_size_inches((7.5,4))
    if save_type:
        fig2e_low.savefig(figure_path + 'Fig2/N2P2_evokedAndTopographies.' + save_type)
    plt.show(block=True)

if create_fig3:
    fig3a = mne.viz.plot_compare_evokeds({'repetition':rep_fz}, picks=['T8'], vlines=[0, 1],
                                 colors={'repetition': color_redpurple},
                                 title='T8-Fz, Blocks {}-{}'.format(start_block, end_block), show=False)
    fig3a[0].axes[0].collections[0].set_lw(0)   #remove outlines of CI
    fig3a[0].axes[0].legend(frameon=False, loc=2)
    fig3a[0].set_size_inches((2,4/3))
    if save_type:
        if end_block==4:
            fig3a[0].savefig(figure_path + 'Fig3/N1_waveform.' + save_type)
        elif end_block==8:
            fig3a[0].savefig(figure_path + 'FigS2/N1_waveform.' + save_type)
        else:
            raise('Other data partitions than 1-4 or 1-8 are currently not supported.')
    plt.show(block=True)

    fig3b = mne.viz.plot_compare_evokeds({'repetition':rep_av}, picks=['Cz'], vlines=[0, 1],
                                 colors={'repetition': color_redpurple},
                                 title='Cz-avg, Blocks {}-{}'.format(start_block, end_block), show=False)
    fig3b[0].axes[0].collections[0].set_lw(0)   #remove outlines of CI
    fig3b[0].axes[0].legend(frameon=False, loc=2)
    fig3b[0].set_size_inches((2,4/3))
    if save_type:
        if end_block==4:
            fig3b[0].savefig(figure_path + 'Fig3/N2P2_waveform.' + save_type)
        elif end_block==8:
            fig3b[0].savefig(figure_path + 'FigS2/N2P2_waveform.' + save_type)
        else:
            raise('Other data partitions than 1-4 or 1-8 are currently not supported.')
    plt.show(block=True)

if get_amplitudes:
    # prepare lists to store peak amplitudes
    sub_list = []
    N1_lat = []
    N2_lat = []
    P2_lat = []
    N1_1st = []
    N1_2nd = []
    N2_1st = []
    N2_2nd = []
    P2_1st = []
    P2_2nd = []
    N1_1st_rep = []
    N1_2nd_rep = []
    N2_1st_rep = []
    N2_2nd_rep = []
    P2_1st_rep = []
    P2_2nd_rep = []
    N1_1st_omi = []
    N1_2nd_omi = []
    N2_1st_omi = []
    N2_2nd_omi = []
    P2_1st_omi = []
    P2_2nd_omi = []
    N1_1st_rep_exp = []
    N1_2nd_rep_exp = []
    N2_1st_rep_exp = []
    N2_2nd_rep_exp = []
    P2_1st_rep_exp = []
    P2_2nd_rep_exp = []
    N1_1st_rep_une = []
    N1_2nd_rep_une = []
    N2_1st_rep_une = []
    N2_2nd_rep_une = []
    P2_1st_rep_une = []
    P2_2nd_rep_une = []
    N1_1st_omi_exp = []
    N1_2nd_omi_exp = []
    N2_1st_omi_exp = []
    N2_2nd_omi_exp = []
    P2_1st_omi_exp = []
    P2_2nd_omi_exp = []
    N1_1st_omi_une = []
    N1_2nd_omi_une = []
    N2_1st_omi_une = []
    N2_2nd_omi_une = []
    P2_1st_omi_une = []
    P2_2nd_omi_une = []

    # find individual peak latencies, check visually, whether those match, extract peak amplitudes for all conditions
    for i_sub, sub in enumerate(subjects):
        _, P2_lat_sub = av[i_sub].copy().pick(['Cz']).get_peak(mode='pos', tmin=(P2_lat_avg - 0.05),
                                                               tmax=(P2_lat_avg + 0.05))
        # for some participants N1 + N2 amplitude is slightly greater than 0, thus mne's get_peak function does not work
        N2_data = av[i_sub].copy().pick(['Cz']).crop(P2_lat_sub-0.2, P2_lat_sub).to_data_frame()
        idx_N2_lat_sub = N2_data.idxmin()["Cz"]
        N2_lat_sub = N2_data.iloc[idx_N2_lat_sub, 0]  # 0 is index of time column

        N1_data = fz[i_sub].copy().pick(['T8']).crop(N2_lat_sub - 0.1, N2_lat_sub).to_data_frame()
        idx_N1_lat_sub = N1_data.idxmin()["T8"]
        N1_lat_sub = N1_data.iloc[idx_N1_lat_sub, 0]  # 0 is index of time column

        mne.viz.plot_compare_evokeds(av[i_sub].copy().crop(0,1),picks=['Cz'],vlines = [N2_lat_sub,P2_lat_sub], title=sub)
        mne.viz.plot_compare_evokeds(fz[i_sub].copy().crop(0,1),picks=['T8'],vlines = [N1_lat_sub], title=sub)

        # if everything looks fine calculate peak amplitudes and store them together with participant and latency
        # information to a file for later analysis in JASP
        sub_list.append(sub)
        N1_lat.append(N1_lat_sub)
        N2_lat.append(N2_lat_sub)
        P2_lat.append(P2_lat_sub)

        N1_1st_rep_exp.append(rep_exp_fz[i_sub].copy().pick(['T8']).crop(N1_lat_sub - 0.015, N1_lat_sub + 0.015).data.mean()*1e6)
        N1_2nd_rep_exp.append(rep_exp_fz[i_sub].copy().pick(['T8']).crop(N1_lat_sub + 1 - 0.015, N1_lat_sub + 1 + 0.015).data.mean()*1e6)
        N2_1st_rep_exp.append(rep_exp_av[i_sub].copy().pick(['Cz']).crop(N2_lat_sub - 0.015, N2_lat_sub + 0.015).data.mean()*1e6)
        N2_2nd_rep_exp.append(rep_exp_av[i_sub].copy().pick(['Cz']).crop(N2_lat_sub + 1 - 0.015, N2_lat_sub + 1 + 0.015).data.mean()*1e6)
        P2_1st_rep_exp.append(rep_exp_av[i_sub].copy().pick(['Cz']).crop(P2_lat_sub - 0.015, P2_lat_sub + 0.015).data.mean()*1e6)
        P2_2nd_rep_exp.append(rep_exp_av[i_sub].copy().pick(['Cz']).crop(P2_lat_sub + 1 - 0.015, P2_lat_sub + 1 + 0.015).data.mean()*1e6)

        N1_1st_rep_une.append(rep_une_fz[i_sub].copy().pick(['T8']).crop(N1_lat_sub - 0.015, N1_lat_sub + 0.015).data.mean() * 1e6)
        N1_2nd_rep_une.append(rep_une_fz[i_sub].copy().pick(['T8']).crop(N1_lat_sub + 1 - 0.015, N1_lat_sub + 1 + 0.015).data.mean() * 1e6)
        N2_1st_rep_une.append(rep_une_av[i_sub].copy().pick(['Cz']).crop(N2_lat_sub - 0.015, N2_lat_sub + 0.015).data.mean()*1e6)
        N2_2nd_rep_une.append(rep_une_av[i_sub].copy().pick(['Cz']).crop(N2_lat_sub + 1 - 0.015, N2_lat_sub + 1 + 0.015).data.mean()*1e6)
        P2_1st_rep_une.append(rep_une_av[i_sub].copy().pick(['Cz']).crop(P2_lat_sub - 0.015, P2_lat_sub + 0.015).data.mean()*1e6)
        P2_2nd_rep_une.append(rep_une_av[i_sub].copy().pick(['Cz']).crop(P2_lat_sub + 1 - 0.015, P2_lat_sub + 1 + 0.015).data.mean()*1e6)

        N1_1st_omi_exp.append(omi_exp_fz[i_sub].copy().pick(['T8']).crop(N1_lat_sub - 0.015, N1_lat_sub + 0.015).data.mean()*1e6)
        N1_2nd_omi_exp.append(omi_exp_fz[i_sub].copy().pick(['T8']).crop(N1_lat_sub + 1 - 0.015, N1_lat_sub + 1 + 0.015).data.mean()*1e6)
        N2_1st_omi_exp.append(omi_exp_av[i_sub].copy().pick(['Cz']).crop(N2_lat_sub - 0.015, N2_lat_sub + 0.015).data.mean()*1e6)
        N2_2nd_omi_exp.append(omi_exp_av[i_sub].copy().pick(['Cz']).crop(N2_lat_sub + 1 - 0.015, N2_lat_sub + 1 + 0.015).data.mean()*1e6)
        P2_1st_omi_exp.append(omi_exp_av[i_sub].copy().pick(['Cz']).crop(P2_lat_sub - 0.015, P2_lat_sub + 0.015).data.mean()*1e6)
        P2_2nd_omi_exp.append(omi_exp_av[i_sub].copy().pick(['Cz']).crop(P2_lat_sub + 1 - 0.015, P2_lat_sub + 1 + 0.015).data.mean()*1e6)

        N1_1st_omi_une.append(omi_une_fz[i_sub].copy().pick(['T8']).crop(N1_lat_sub - 0.015, N1_lat_sub + 0.015).data.mean()*1e6)
        N1_2nd_omi_une.append(omi_une_fz[i_sub].copy().pick(['T8']).crop(N1_lat_sub + 1 - 0.015, N1_lat_sub + 1 + 0.015).data.mean()*1e6)
        N2_1st_omi_une.append(omi_une_av[i_sub].copy().pick(['Cz']).crop(N2_lat_sub - 0.015, N2_lat_sub + 0.015).data.mean()*1e6)
        N2_2nd_omi_une.append(omi_une_av[i_sub].copy().pick(['Cz']).crop(N2_lat_sub + 1 - 0.015, N2_lat_sub + 1 + 0.015).data.mean()*1e6)
        P2_1st_omi_une.append(omi_une_av[i_sub].copy().pick(['Cz']).crop(P2_lat_sub - 0.015, P2_lat_sub + 0.015).data.mean()*1e6)
        P2_2nd_omi_une.append(omi_une_av[i_sub].copy().pick(['Cz']).crop(P2_lat_sub + 1 - 0.015, P2_lat_sub + 1 + 0.015).data.mean()*1e6)

        N1_1st_rep.append(rep_fz[i_sub].copy().pick(['T8']).crop(N1_lat_sub - 0.015, N1_lat_sub + 0.015).data.mean() * 1e6)
        N1_2nd_rep.append(rep_fz[i_sub].copy().pick(['T8']).crop(N1_lat_sub + 1 - 0.015, N1_lat_sub + 1 + 0.015).data.mean() * 1e6)
        N2_1st_rep.append(rep_av[i_sub].copy().pick(['Cz']).crop(N2_lat_sub - 0.015, N2_lat_sub + 0.015).data.mean()*1e6)
        N2_2nd_rep.append(rep_av[i_sub].copy().pick(['Cz']).crop(N2_lat_sub + 1 - 0.015, N2_lat_sub + 1 + 0.015).data.mean()*1e6)
        P2_1st_rep.append(rep_av[i_sub].copy().pick(['Cz']).crop(P2_lat_sub - 0.015, P2_lat_sub + 0.015).data.mean()*1e6)
        P2_2nd_rep.append(rep_av[i_sub].copy().pick(['Cz']).crop(P2_lat_sub + 1 - 0.015, P2_lat_sub + 1 + 0.015).data.mean()*1e6)

        N1_1st_omi.append(omi_fz[i_sub].copy().pick(['T8']).crop(N1_lat_sub - 0.015, N1_lat_sub + 0.015).data.mean() * 1e6)
        N1_2nd_omi.append(omi_fz[i_sub].copy().pick(['T8']).crop(N1_lat_sub + 1 - 0.015, N1_lat_sub + 1 + 0.015).data.mean() * 1e6)
        N2_1st_omi.append(omi_av[i_sub].copy().pick(['Cz']).crop(N2_lat_sub - 0.015, N2_lat_sub + 0.015).data.mean()*1e6)
        N2_2nd_omi.append(omi_av[i_sub].copy().pick(['Cz']).crop(N2_lat_sub + 1 - 0.015, N2_lat_sub + 1 + 0.015).data.mean()*1e6)
        P2_1st_omi.append(omi_av[i_sub].copy().pick(['Cz']).crop(P2_lat_sub - 0.015, P2_lat_sub + 0.015).data.mean()*1e6)
        P2_2nd_omi.append(omi_av[i_sub].copy().pick(['Cz']).crop(P2_lat_sub + 1 - 0.015, P2_lat_sub + 1 + 0.015).data.mean()*1e6)

        N1_1st.append(fz[i_sub].copy().pick(['T8']).crop(N1_lat_sub - 0.015, N1_lat_sub + 0.015).data.mean() * 1e6)
        N1_2nd.append(fz[i_sub].copy().pick(['T8']).crop(N1_lat_sub + 1 - 0.015,N1_lat_sub + 1 + 0.015).data.mean() * 1e6)
        N2_1st.append(av[i_sub].copy().pick(['Cz']).crop(N2_lat_sub - 0.015, N2_lat_sub + 0.015).data.mean()*1e6)
        N2_2nd.append(av[i_sub].copy().pick(['Cz']).crop(N2_lat_sub + 1 - 0.015, N2_lat_sub + 1 + 0.015).data.mean()*1e6)
        P2_1st.append(av[i_sub].copy().pick(['Cz']).crop(P2_lat_sub - 0.015, P2_lat_sub + 0.015).data.mean()*1e6)
        P2_2nd.append(av[i_sub].copy().pick(['Cz']).crop(P2_lat_sub + 1 - 0.015, P2_lat_sub + 1 + 0.015).data.mean()*1e6)

    amplitudes = pd.DataFrame({'subject':sub_list, 'N1_latency':N1_lat, 'N2_latency':N2_lat, 'P2_latency':P2_lat,
                               '1st_N1_rep_exp':N1_1st_rep_exp, '2nd_N1_rep_exp':N1_2nd_rep_exp,
                               '1st_N2_rep_exp':N2_1st_rep_exp, '2nd_N2_rep_exp':N2_2nd_rep_exp,
                               '1st_P2_rep_exp':P2_1st_rep_exp, '2nd_P2_rep_exp':P2_2nd_rep_exp,
                               '1st_N1_rep_une':N1_1st_rep_une, '2nd_N1_rep_une':N1_2nd_rep_une,
                               '1st_N2_rep_une':N2_1st_rep_une, '2nd_N2_rep_une':N2_2nd_rep_une,
                               '1st_P2_rep_une':P2_1st_rep_une, '2nd_P2_rep_une':P2_2nd_rep_une,
                               '1st_N1_omi_exp':N1_1st_omi_exp, '2nd_N1_omi_exp':N1_2nd_omi_exp,
                               '1st_N2_omi_exp':N2_1st_omi_exp, '2nd_N2_omi_exp':N2_2nd_omi_exp,
                               '1st_P2_omi_exp':P2_1st_omi_exp, '2nd_P2_omi_exp':P2_2nd_omi_exp,
                               '1st_N1_omi_une':N1_1st_omi_une, '2nd_N1_omi_une':N1_2nd_omi_une,
                               '1st_N2_omi_une':N2_1st_omi_une, '2nd_N2_omi_une':N2_2nd_omi_une,
                               '1st_P2_omi_une':P2_1st_omi_une, '2nd_P2_omi_une':P2_2nd_omi_une,
                               '1st_N1_rep':N1_1st_rep, '2nd_N1_rep':N1_2nd_rep, '1st_N2_rep':N2_1st_rep,
                               '2nd_N2_rep':N2_2nd_rep, '1st_P2_rep':P2_1st_rep, '2nd_P2_rep':P2_2nd_rep,
                               '1st_N1_omi':N1_1st_omi, '2nd_N1_omi':N1_2nd_omi, '1st_N2_omi':N2_1st_omi,
                               '2nd_N2_omi':N2_2nd_omi, '1st_P2_omi':P2_1st_omi, '2nd_P2_omi':P2_2nd_omi,
                               '1st_N1':N1_1st, '2nd_N1':N1_2nd, '1st_N2':N2_1st, '2nd_N2':N2_2nd, '1st_P2':P2_1st,
                               '2nd_P2':P2_2nd})
    amplitudes["1st_N2P2_rep_exp"] = amplitudes["1st_P2_rep_exp"] - amplitudes["1st_N2_rep_exp"]
    amplitudes["2nd_N2P2_rep_exp"] = amplitudes["2nd_P2_rep_exp"] - amplitudes["2nd_N2_rep_exp"]
    amplitudes["1st_N2P2_rep_une"] = amplitudes["1st_P2_rep_une"] - amplitudes["1st_N2_rep_une"]
    amplitudes["2nd_N2P2_rep_une"] = amplitudes["2nd_P2_rep_une"] - amplitudes["2nd_N2_rep_une"]
    amplitudes["1st_N2P2_omi_exp"] = amplitudes["1st_P2_omi_exp"] - amplitudes["1st_N2_omi_exp"]
    amplitudes["2nd_N2P2_omi_exp"] = amplitudes["2nd_P2_omi_exp"] - amplitudes["2nd_N2_omi_exp"]
    amplitudes["1st_N2P2_omi_une"] = amplitudes["1st_P2_omi_une"] - amplitudes["1st_N2_omi_une"]
    amplitudes["2nd_N2P2_omi_une"] = amplitudes["2nd_P2_omi_une"] - amplitudes["2nd_N2_omi_une"]
    amplitudes["1st_N2P2_rep"] = amplitudes["1st_P2_rep"] - amplitudes["1st_N2_rep"]
    amplitudes["2nd_N2P2_rep"] = amplitudes["2nd_P2_rep"] - amplitudes["2nd_N2_rep"]
    amplitudes["1st_N2P2_omi"] = amplitudes["1st_P2_omi"] - amplitudes["1st_N2_omi"]
    amplitudes["2nd_N2P2_omi"] = amplitudes["2nd_P2_omi"] - amplitudes["2nd_N2_omi"]
    amplitudes.to_csv(os.path.join(result_path, 'amplitudes_N2P2_blocks{}-{}.csv'.format(start_block, end_block)), index=False, na_rep='n/a')

if create_fig4:
    fig4a = mne.viz.plot_compare_evokeds({'expected':rep_exp_fz, 'unexpected':rep_une_fz}, picks=['T8'],
                                 colors={'expected': color_red, 'unexpected': color_purple},
                                 title='Grand average, repetitions, T8-Fz',vlines=[0,1], show=False)
    fig4a[0].axes[0].collections[0].set_lw(0)   #remove outlines of CI
    fig4a[0].axes[0].collections[1].set_lw(0)
    fig4a[0].axes[0].legend(frameon=False, loc=2)
    fig4a[0].set_size_inches((2,4/3))
    if save_type:
        if end_block==4:
            fig4a[0].savefig(figure_path + 'Fig4/N1_waveform.' + save_type)
        elif end_block==8:
            fig4a[0].savefig(figure_path + 'FigS3/N1_waveform.' + save_type)
        else:
            raise('Other data partitions than 1-4 or 1-8 are currently not supported.')
    plt.show(block=True)

    fig4b = mne.viz.plot_compare_evokeds({'expected':rep_exp_av, 'unexpected':rep_une_av}, picks=['Cz'],
                                 colors={'expected': color_red, 'unexpected': color_purple},
                                 title='Grand average, repetitions, Cz-average',vlines=[0,1], show=False)
    fig4b[0].axes[0].collections[0].set_lw(0)   #remove outlines of CI
    fig4b[0].axes[0].collections[1].set_lw(0)
    fig4b[0].axes[0].legend(frameon=False, loc=2)
    fig4b[0].set_size_inches((2,4/3))
    if save_type:
        if end_block==4:
            fig4b[0].savefig(figure_path + 'Fig4/N2P2_waveform.' + save_type)
        elif end_block==8:
            fig4b[0].savefig(figure_path + 'FigS3/N2P2_waveform.' + save_type)
        else:
            raise('Other data partitions than 1-4 or 1-8 are currently not supported.')
    plt.show(block=True)

if create_figS1:
    figS1a = mne.viz.plot_compare_evokeds({'expected':omi_exp_fz, 'unexpected':omi_une_fz}, picks=['T8'],
                                 colors={'expected': color_green, 'unexpected': color_blue},
                                 title='Grand average, omissions, T8-Fz',vlines=[0,1], show=False)
    figS1a[0].axes[0].collections[0].set_lw(0)   #remove outlines of CI
    figS1a[0].axes[0].collections[1].set_lw(0)
    figS1a[0].axes[0].legend(frameon=False, loc=2)
    figS1a[0].set_size_inches((2,4/3))
    if save_type:
        if end_block==4:
            figS1a[0].savefig(figure_path + 'FigS1/N1_waveform.' + save_type)
        elif end_block==8:
            figS1a[0].savefig(figure_path + 'FigS4/N1_waveform.' + save_type)
        else:
            raise('Other data partitions than 1-4 or 1-8 are currently not supported.')
    plt.show(block=True)

    figS1b = mne.viz.plot_compare_evokeds({'expected':omi_exp_av, 'unexpected':omi_une_av}, picks=['Cz'],
                                 colors={'expected': color_green, 'unexpected': color_blue},
                                 title='Grand average, omissions, Cz-average',vlines=[0,1], show=False)
    figS1b[0].axes[0].collections[0].set_lw(0)   #remove outlines of CI
    figS1b[0].axes[0].collections[1].set_lw(0)
    figS1b[0].axes[0].legend(frameon=False, loc=2)
    figS1b[0].set_size_inches((2,4/3))
    if save_type:
        if end_block==4:
            figS1b[0].savefig(figure_path + 'FigS1/N2P2_waveform.' + save_type)
        elif end_block==8:
            figS1b[0].savefig(figure_path + 'FigS4/N2P2_waveform.' + save_type)
        else:
            raise('Other data partitions than 1-4 or 1-8 are currently not supported.')
    plt.show(block=True)

if cbpt:
    adjacency, ch_names = find_ch_adjacency(rep_av[0].info, "eeg")
    
    ####################################
    # CBPT for repetitions vs. omissions
    ####################################
    reps = np.dstack([rep_av[i].get_data() for i in range(36)]).transpose(2, 1, 0)
    omis = np.dstack([omi_av[i].get_data() for i in range(36)]).transpose(2, 1, 0)
    X = [reps, omis]
	
    # CBPT
    cluster_stats = spatio_temporal_cluster_test(
        X,
        n_permutations=10000,
        adjacency=adjacency,
    )
    t_obs, clusters, p_values, _ = cluster_stats

    p_accept = 0.05
    good_cluster_inds = np.where(p_values < p_accept)[0]

    # preparation for plotting
    colors = {"repetition": "crimson", "omission": "steelblue"}
    evokeds = {'repetition':mne.grand_average(rep_av), 'omission':mne.grand_average(omi_av)}

    # loop over clusters
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        # unpack cluster information, get unique indices
        time_inds, space_inds = np.squeeze(clusters[clu_idx])
        ch_inds = np.unique(space_inds)
        time_inds = np.unique(time_inds)

        # get topography for F stat
        t_map = t_obs[time_inds, ...].mean(axis=0)

        # get times from samples
        sig_times = rep_av[0].times[time_inds]

        # create spatial mask
        mask = np.zeros((t_map.shape[0], 1), dtype=bool)
        mask[ch_inds, :] = True

        # initialize figure
        fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3), layout="constrained")

        # plot average test statistic and mark significant sensors
        t_evoked = mne.EvokedArray(t_map[:, np.newaxis], rep_av[0].info, tmin=0)
        t_evoked.plot_topomap(
            times=0,
            mask=mask,
            axes=ax_topo,
            cmap="Reds",
            vlim=(np.min, np.max),
            show=False,
            colorbar=False,
            mask_params=dict(markersize=10),
        )
        image = ax_topo.images[0]
        ax_topo.set_title("")

        # create additional axes (for ERF and colorbar)
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax_topo)

        # add axes for colorbar
        ax_colorbar = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(image, cax=ax_colorbar)
        ax_topo.set_xlabel(
            "Averaged t-map ({:0.3f} - {:0.3f} s)".format(*sig_times[[0, -1]])
        )

        # add new axis for time courses and plot time courses
        ax_signals = divider.append_axes("right", size="300%", pad=1.2)
        title = f"Cluster #{i_clu + 1}, {len(ch_inds)} sensor"
        if len(ch_inds) > 1:
            title += "s (mean)"
        mne.viz.plot_compare_evokeds(
            evokeds,
            title=title,
            picks=ch_inds,
            axes=ax_signals,
            colors=colors,
            show=False,
            split_legend=True,
            truncate_yaxis="auto",
        )

        # plot temporal cluster extent
        ymin, ymax = ax_signals.get_ylim()
        ax_signals.fill_betweenx(
            (ymin, ymax), sig_times[0], sig_times[-1], color="orange", alpha=0.3
        )

    plt.show(block=True)

    if len(good_cluster_inds) == 0:
        print('Repetition vs. Omission: No significant clusters found!')
    
    ##############################################
    # CBPT for expected vs. unexpected repetitions
    ##############################################
    exp = np.dstack([rep_exp_av[i].get_data() for i in range(36)]).transpose(2, 1, 0)
    une = np.dstack([rep_une_av[i].get_data() for i in range(36)]).transpose(2, 1, 0)
    X = [exp, une]
	
    # CBPT
    cluster_stats = spatio_temporal_cluster_test(
        X,
        n_permutations=10000,
        adjacency=adjacency,
    )
    t_obs, clusters, p_values, _ = cluster_stats

    p_accept = 0.05
    good_cluster_inds = np.where(p_values < p_accept)[0]

    # preparation for plotting
    colors = {"expected": "crimson", "unexpected": "steelblue"}
    evokeds = {'expected':mne.grand_average(rep_exp_av), 'unexpected':mne.grand_average(rep_une_av)}

    # loop over clusters
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        # unpack cluster information, get unique indices
        time_inds, space_inds = np.squeeze(clusters[clu_idx])
        ch_inds = np.unique(space_inds)
        time_inds = np.unique(time_inds)

        # get topography for F stat
        t_map = t_obs[time_inds, ...].mean(axis=0)

        # get times from samples
        sig_times = rep_av[0].times[time_inds]

        # create spatial mask
        mask = np.zeros((t_map.shape[0], 1), dtype=bool)
        mask[ch_inds, :] = True

        # initialize figure
        fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3), layout="constrained")

        # plot average test statistic and mark significant sensors
        t_evoked = mne.EvokedArray(t_map[:, np.newaxis], rep_av[0].info, tmin=0)
        t_evoked.plot_topomap(
            times=0,
            mask=mask,
            axes=ax_topo,
            cmap="Reds",
            vlim=(np.min, np.max),
            show=False,
            colorbar=False,
            mask_params=dict(markersize=10),
        )
        image = ax_topo.images[0]
        ax_topo.set_title("")

        # create additional axes (for ERF and colorbar)
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax_topo)

        # add axes for colorbar
        ax_colorbar = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(image, cax=ax_colorbar)
        ax_topo.set_xlabel(
            "Averaged t-map ({:0.3f} - {:0.3f} s)".format(*sig_times[[0, -1]])
        )

        # add new axis for time courses and plot time courses
        ax_signals = divider.append_axes("right", size="300%", pad=1.2)
        title = f"Cluster #{i_clu + 1}, {len(ch_inds)} sensor"
        if len(ch_inds) > 1:
            title += "s (mean)"
        mne.viz.plot_compare_evokeds(
            evokeds,
            title=title,
            picks=ch_inds,
            axes=ax_signals,
            colors=colors,
            show=False,
            split_legend=True,
            truncate_yaxis="auto",
        )

        # plot temporal cluster extent
        ymin, ymax = ax_signals.get_ylim()
        ax_signals.fill_betweenx(
            (ymin, ymax), sig_times[0], sig_times[-1], color="orange", alpha=0.3
        )

    plt.show(block=True)

    if len(good_cluster_inds) == 0:
        print('Expected vs. Unexpected Repetitions: No significant clusters found!')

    ############################################
    # CBPT for expected vs. unexpected omissions
    ############################################
    exp = np.dstack([omi_exp_av[i].get_data() for i in range(36)]).transpose(2, 1, 0)
    une = np.dstack([omi_une_av[i].get_data() for i in range(36)]).transpose(2, 1, 0)
    X = [exp, une]
	
    # CBPT
    cluster_stats = spatio_temporal_cluster_test(
        X,
        n_permutations=10000,
        adjacency=adjacency,
    )
    t_obs, clusters, p_values, _ = cluster_stats

    p_accept = 0.05
    good_cluster_inds = np.where(p_values < p_accept)[0]

    # preparation for plotting
    colors = {"expected": "crimson", "unexpected": "steelblue"}
    evokeds = {'expected':mne.grand_average(omi_exp_av), 'unexpected':mne.grand_average(omi_une_av)}

    # loop over clusters
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        # unpack cluster information, get unique indices
        time_inds, space_inds = np.squeeze(clusters[clu_idx])
        ch_inds = np.unique(space_inds)
        time_inds = np.unique(time_inds)

        # get topography for F stat
        t_map = t_obs[time_inds, ...].mean(axis=0)

        # get times from samples
        sig_times = omi_av[0].times[time_inds]

        # create spatial mask
        mask = np.zeros((t_map.shape[0], 1), dtype=bool)
        mask[ch_inds, :] = True

        # initialize figure
        fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3), layout="constrained")

        # plot average test statistic and mark significant sensors
        t_evoked = mne.EvokedArray(t_map[:, np.newaxis], omi_av[0].info, tmin=0)
        t_evoked.plot_topomap(
            times=0,
            mask=mask,
            axes=ax_topo,
            cmap="Reds",
            vlim=(np.min, np.max),
            show=False,
            colorbar=False,
            mask_params=dict(markersize=10),
        )
        image = ax_topo.images[0]
        ax_topo.set_title("")

        # create additional axes (for ERF and colorbar)
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax_topo)

        # add axes for colorbar
        ax_colorbar = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(image, cax=ax_colorbar)
        ax_topo.set_xlabel(
            "Averaged t-map ({:0.3f} - {:0.3f} s)".format(*sig_times[[0, -1]])
        )

        # add new axis for time courses and plot time courses
        ax_signals = divider.append_axes("right", size="300%", pad=1.2)
        title = f"Cluster #{i_clu + 1}, {len(ch_inds)} sensor"
        if len(ch_inds) > 1:
            title += "s (mean)"
        mne.viz.plot_compare_evokeds(
            evokeds,
            title=title,
            picks=ch_inds,
            axes=ax_signals,
            colors=colors,
            show=False,
            split_legend=True,
            truncate_yaxis="auto",
        )

        # plot temporal cluster extent
        ymin, ymax = ax_signals.get_ylim()
        ax_signals.fill_betweenx(
            (ymin, ymax), sig_times[0], sig_times[-1], color="orange", alpha=0.3
        )

    plt.show(block=True)

    if len(good_cluster_inds) == 0:
        print('Expected vs. Unexpected Omissions: No significant clusters found!')
