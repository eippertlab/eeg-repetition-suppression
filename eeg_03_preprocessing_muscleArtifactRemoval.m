%%%
% preprocessing of eeg data - removes muscle artifacts
% ----------------------------------------------------
% removes spike-artefacts from an EEG-trace as described in
% Liebisch A. P., Eggert T., Valentini E., Irving S., Schulz E. (2020)
% uses functions EegCleanLib.m, fit_1dGabor.m, set_default_parameters.m,
% findmaxi.m and deal_num.m
% (can all be found under
% https://drive.google.com/file/d/1_-O2No8Knm263IrniZSSoawH7ZN3gxeb/view
% or navigate from https://www.pain.sc/publications to the software download)
%
% authors:
% --------
% Lisa-Marie Pohle, based on example script (https://drive.google.com/file/d/1_-O2No8Knm263IrniZSSoawH7ZN3gxeb/view)
%
% contact:
% --------
% lmpohle@cbs.mpg.de
%
% date:
% -----
% 5th June 2025
%%%

clear;
close all;

subjects = {'sub-es01','sub-es02','sub-es03','sub-es04','sub-es05', ...
    'sub-es06','sub-es07','sub-es08','sub-es09','sub-es10','sub-es11', ...
    'sub-es12','sub-es13','sub-es14','sub-es15','sub-es16','sub-es17', ...
    'sub-es18','sub-es19','sub-es20','sub-es21','sub-es22','sub-es23', ...
    'sub-es24','sub-es25','sub-es26','sub-es27','sub-es28','sub-es29', ...
    'sub-es30','sub-es31','sub-es32','sub-es33','sub-es34','sub-es35', ...
    'sub-es37','sub-es38','sub-es39','sub-es40','sub-es41','sub-es42', ...
    'sub-es43','sub-es44','sub-es45','sub-es46','sub-es47'};

data_path = '/data/pt_02299/ExpSupp/Main/derivatives';
file_name_suffix_in = '_woAborts_hpfilter_manReject_icspace_epo.mat';
file_name_suffix_out = '_woAborts_hpfilter_manReject_muscleCleaned_icspace_epo.mat';
% where the function EegCleanLib.m and extra utility functions are located:
tool_function_path = '/data/pt_02299/EEG_Spike_artefact_removal_tool';
addpath(genpath(tool_function_path));

SamplingRate = 500;

clean_options = [];
clean_options.median_filter_duration_s = 0.1;   % window width for the running median high-pass filter 
clean_options.min_tpeak_diff_s         = 0.015; % peaks which are closer than min_tpeak_diff_s [s] to another peak, are assumed to belong to the same spike
clean_options.cutoff_window_width_s    = 0.2;   % gabor-fitting of the spike is done within this window
clean_options.fignr                    = -1;    % 0: automatic figure number; <0: no plot 
clean_options.individual_fit_fignr     = -1;    % 0: automatic figure number; <0: no plot
clean_options.cluster_plot_fignr       = -1;    % 0: automatic figure number; <0: no plot
clean_options.no_print                 = true;
clean_options.max_event_width_s        = 0.025; % maximal width of an event (+- 3rd zerocrossing) 
clean_options.run_count                = 2;
clean_options.analyse_window_width_s   = 0.5;  % detection of peaks is done within a running window with analyse_window_width_s in s. 

for isub = 1:numel(subjects)
    sub = subjects{isub};
    
    % load data and prepare data structure (data + time matrices), data has dimensions N_epochs (up to 192) x N_ICs (32) x N_timepoints (3000)
    data = load(fullfile(data_path, sub, 'eeg', [sub file_name_suffix_in]));
    d_old_all = data.d_old_all;
    t = (0:size(d_old_all,3)-1)/SamplingRate-2; %subtraction of 2, since interval starts at -2s relative to onset of the first stimulus
    
    ecl=EegCleanLib();
    
    d_clean_all = zeros(size(d_old_all));
    
    % run cleaning for all epochs and ICs
    for i_ic = 1:size(d_old_all, 2)
        parfor i_epo = 1:size(d_old_all, 1)
            disp(['Subject = '  sub  ', IC = ' num2str(i_ic) ', Epoch = ' num2str(i_epo)])
            d_old = d_old_all(i_epo, i_ic, :);
            d_old = squeeze(d_old);
            [d_clean,fitresults]=ecl.clean_spikes(d_old, SamplingRate, clean_options);
            d_clean_all(i_epo, i_ic, :) = d_clean;
        end
    end

    saveCleanData(data_path, sub, d_clean_all, file_name_suffix_out)
    disp(['################# ' sub ' done! #####################'])
end

function saveCleanData(path, subject, variable, suffix)
    save(fullfile(path, subject, 'eeg', [subject suffix]), 'variable')
end
