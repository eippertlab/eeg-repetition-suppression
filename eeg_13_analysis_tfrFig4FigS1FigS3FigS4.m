%%%
% analysis of eeg data - time-frequency domain, expected vs. unexpected trials
% ----------------------------------------------------------------------------
% load data per condition
% subtractive baseline correction
% average over repetitions per participant
% extract mean power at Cz in prespecified time-frequency bins
% save values for later analysis in JASP
% perform cluster-based permutation test between expected and unexpected trials
% can be run for repetition OR omission trials
% 
% Fieldtrip functions need to be added to the Matlab path
% we used Fieldtrip version 0.20220623
%
% authors:
% --------
% Lisa-Marie Pohle
%
% contact:
% --------
% lmpohle@cbs.mpg.de
%
% date:
% -----
% 5th June 2025
%%%

subjects = {'sub-es01', 'sub-es03', 'sub-es04', 'sub-es06', 'sub-es07', ...
    'sub-es08', 'sub-es09', 'sub-es10', 'sub-es11', 'sub-es12', ...
    'sub-es13', 'sub-es15', 'sub-es17', 'sub-es18', 'sub-es19', ...
    'sub-es20', 'sub-es21', 'sub-es22', 'sub-es23', 'sub-es25', ...
    'sub-es26', 'sub-es27', 'sub-es28', 'sub-es30', 'sub-es31', ...
    'sub-es32', 'sub-es33', 'sub-es34', 'sub-es35', 'sub-es40', ...
    'sub-es41', 'sub-es42', 'sub-es44', 'sub-es45', 'sub-es46', 'sub-es47'};

chosen_fig = 'fig4';

switch chosen_fig
    case 'fig4' 
        task1 = 'repetition/expected';
        task2 = 'repetition/unexpected';
        used_blocks = '14';
        roi_values = table('Size',[36 13], ...
            'VariableTypes',["string","double","double","double", ...
            "double","double","double","double","double","double", ...
            "double","double","double"], ...
            'VariableNames',["Subject", "alpha1_rep_exp", ...
            "alpha2_rep_exp", "beta1_rep_exp", "beta2_rep_exp", ...
            "gamma1_rep_exp", "gamma2_rep_exp", "alpha1_rep_une", ...
            "alpha2_rep_une", "beta1_rep_une", "beta2_rep_une", ...
            "gamma1_rep_une", "gamma2_rep_une"]);
        folder = 'Fig4';
    case 'figS1'
        task1 = 'omission/expected';
        task2 = 'omission/unexpected';
        used_blocks = '14';
        roi_values = table('Size',[36 13], ...
            'VariableTypes',["string","double","double","double", ...
            "double","double","double","double","double","double", ...
            "double","double","double"], ...
            'VariableNames',["Subject", "alpha1_omi_exp", ...
            "alpha2_omi_exp", "beta1_omi_exp", "beta2_omi_exp", ...
            "gamma1_omi_exp", "gamma2_omi_exp", "alpha1_omi_une", ...
            "alpha2_omi_une", "beta1_omi_une", "beta2_omi_une", ...
            "gamma1_omi_une", "gamma2_omi_une"]);
        folder = 'FigS1';
    case 'figS3'
        task1 = 'repetition/expected';
        task2 = 'repetition/unexpected';
        used_blocks = '18';
        roi_values = table('Size',[36 13], ...
            'VariableTypes',["string","double","double","double", ...
            "double","double","double","double","double","double", ...
            "double","double","double"], ...
            'VariableNames',["Subject", "alpha1_rep_exp", ...
            "alpha2_rep_exp", "beta1_rep_exp", "beta2_rep_exp", ...
            "gamma1_rep_exp", "gamma2_rep_exp", "alpha1_rep_une", ...
            "alpha2_rep_une", "beta1_rep_une", "beta2_rep_une", ...
            "gamma1_rep_une", "gamma2_rep_une"]);
        folder = 'FigS3';
    case 'figS4'
        task1 = 'omission/expected';
        task2 = 'omission/unexpected';
        used_blocks = '18';
        roi_values = table('Size',[36 13], ...
            'VariableTypes',["string","double","double","double", ...
            "double","double","double","double","double","double", ...
            "double","double","double"], ...
            'VariableNames',["Subject", "alpha1_omi_exp", ...
            "alpha2_omi_exp", "beta1_omi_exp", "beta2_omi_exp", ...
            "gamma1_omi_exp", "gamma2_omi_exp", "alpha1_omi_une", ...
            "alpha2_omi_une", "beta1_omi_une", "beta2_omi_une", ...
            "gamma1_omi_une", "gamma2_omi_une"]);
        folder = 'FigS4';
    otherwise
        error('Unexpected figure number. Choose from fig4, figS1, figS3, figS4')
end

layout_file  = '/software/fieldtrip/0.20220623/debian-bullseye-amd64/source/template/layout/easycapM24.mat';
elec = ft_read_sens('/software/fieldtrip/0.20220623/debian-bullseye-amd64/source/template/electrode/easycap-M1.txt');

subset_task1 = cell(1,numel(subjects));
subset_task2 = cell(1,numel(subjects));

for isub = 1:numel(subjects)
    sub = subjects{isub};

    load(['/data/pt_02299/ExpSupp/Main/derivatives/' sub '/tfr/' sub '_task-expsupp_tfr.mat'])
    events = readtable(['/data/pt_02299/ExpSupp/Main/derivatives/' sub '/eeg/' sub '_woAborts_hpfilter_manReject_muscleCleaned_ICsRejected_autoReject50_manReject_equalizedCounts_events.tsv'], "FileType","delimitedtext");
    index_toDelete = find(strcmp(events.(['drop_log_blocks' used_blocks]), '(''MAN REJECT POST ICA'',)'));
    index_toDelete = cat(1, index_toDelete, find(strcmp(events.(['drop_log_blocks' used_blocks]), '(''AUTO REJECT POST ICA'',)')));
    index_toDelete = cat(1, index_toDelete, find(strcmp(events.(['drop_log_blocks' used_blocks]), '(''MAN REJECT PRE ICA'',)')));
    index_toDelete = cat(1, index_toDelete, find(strcmp(events.(['drop_log_blocks' used_blocks]), '(''USER'',)')));
    index_toDelete = cat(1, index_toDelete, find(strcmp(events.(['drop_log_blocks' used_blocks]), '(''INVALID TRIAL'',)')));
    index_toDelete = cat(1, index_toDelete, find(strcmp(events.(['drop_log_blocks' used_blocks]), '(''LASER ABORT'',)')));
    events(index_toDelete,:) = [];

    pattern = ['1st/' task1];
    index1_subset = find(strcmp(events.(['drop_log_blocks' used_blocks]), '()') & contains(events.trial_type, pattern));
    pattern = ['1st/' task2];
    index2_subset = find(strcmp(events.(['drop_log_blocks' used_blocks]), '()') & contains(events.trial_type, pattern));
    
    cfg = [];
    cfg.avgoverrpt = 'no';
    cfg.trials = index1_subset;
    subset_task1_epochs = ft_selectdata(cfg, tfr);
    cfg.trials = index2_subset;
    subset_task2_epochs = ft_selectdata(cfg, tfr);

    cfg = [];
    cfg.parameter = 'powspctrm';
    cfg.baseline = [-0.75, -0.25];
    cfg.baselinetype   = 'absolute';
    subset_task1_epochs_base = ft_freqbaseline(cfg, subset_task1_epochs);
    subset_task2_epochs_base = ft_freqbaseline(cfg, subset_task2_epochs);

    cfg = [];
    cfg.avgoverrpt = 'yes';
    cfg.trial = 'all';
    subset_task1{isub} = ft_selectdata(cfg, subset_task1_epochs_base);
    subset_task2{isub} = ft_selectdata(cfg, subset_task2_epochs_base);

    cfg = [];
    cfg.avgoverrpt = 'yes';
    cfg.trials = 'all';
    cfg.channel = 'Cz';
    subset_for_roi_task1 = ft_selectdata(cfg, subset_task1_epochs_base);
    subset_for_roi_task2 = ft_selectdata(cfg, subset_task2_epochs_base);

    roi_values(isub, 1) = {sub};
    roi_values(isub, 2) = {mean(subset_for_roi_task1.powspctrm(:,8:12,76:100), 'all')};
    roi_values(isub, 3) = {mean(subset_for_roi_task1.powspctrm(:,8:12,126:150), 'all')};
    roi_values(isub, 4) = {mean(subset_for_roi_task1.powspctrm(:,13:30,66:85), 'all')};
    roi_values(isub, 5) = {mean(subset_for_roi_task1.powspctrm(:,13:30,116:135), 'all')};
    roi_values(isub, 6) = {mean(subset_for_roi_task1.powspctrm(:,70:90,58:73), 'all')};
    roi_values(isub, 7) = {mean(subset_for_roi_task1.powspctrm(:,70:90,108:123), 'all')};
    roi_values(isub, 8) = {mean(subset_for_roi_task2.powspctrm(:,8:12,76:100), 'all')};
    roi_values(isub, 9) = {mean(subset_for_roi_task2.powspctrm(:,8:12,126:150), 'all')};
    roi_values(isub, 10) = {mean(subset_for_roi_task2.powspctrm(:,13:30,66:85), 'all')};
    roi_values(isub, 11) = {mean(subset_for_roi_task2.powspctrm(:,13:30,116:135), 'all')};
    roi_values(isub, 12) = {mean(subset_for_roi_task2.powspctrm(:,70:90,58:73), 'all')};
    roi_values(isub, 13) = {mean(subset_for_roi_task2.powspctrm(:,70:90,108:123), 'all')};

    subset_task1{isub}.elec = elec;
    subset_task2{isub}.elec = elec;

    disp(['################# ' sub ' loaded! #####################'])
end

%writetable(roi_values,['/data/pt_02299/ExpSupp/Main/results/eeg/' chosen_fig '_tfr_rois.csv'])

cfg = [];
cfg.keepindividual = 'yes';
grandavg_task1 = ft_freqgrandaverage(cfg, subset_task1{:});
grandavg_task2 = ft_freqgrandaverage(cfg, subset_task2{:});
grandavg_task1.elec = elec;

cfg = [];
cfg.channel          = {'all'};
cfg.latency          = 'all';
cfg.method           = 'montecarlo';
cfg.statistic        = 'ft_statfun_depsamplesT';
cfg.correctm         = 'cluster';
cfg.clusteralpha     = 0.05;
cfg.clusterstatistic = 'maxsum';
cfg.minnbchan        = 2;
cfg.tail             = 0;
cfg.clustertail      = 0;
cfg.alpha            = 0.025;
cfg.numrandomization = 10000;
%prepare_neighbours determines what sensors may form clusters
cfg_neighb.method    = 'distance';
cfg_neighb.channel = 'all';
cfg_neighb.neighbourdist = 60;
cfg.neighbours       = ft_prepare_neighbours(cfg_neighb, grandavg_task1);

design = [ones(1,size(grandavg_task1.powspctrm,1)), ones(1,size(grandavg_task1.powspctrm,1))*2;
    1:size(grandavg_task1.powspctrm,1), 1:size(grandavg_task1.powspctrm,1)];

cfg.design           = design;
cfg.ivar             = 1;
cfg.uvar             = 2;

[stat] = ft_freqstatistics(cfg, grandavg_task1, grandavg_task2);

%for omission trials omit es41 (no 31) for plotting
cfg = [];
switch chosen_fig
    case {'fig4', 'figS3'}
        cfg.trials = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,...
            20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]; 
    case {'figS1', 'figS4'}
        cfg.trials = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,...
            20,21,22,23,24,25,26,27,28,29,30,32,33,34,35,36]; 
    otherwise
        error('Unexpected figure number. Choose from fig4, figS1, figS3, figS4')
end
avg_task1 = ft_freqdescriptives(cfg, grandavg_task1);
avg_task2 = ft_freqdescriptives(cfg, grandavg_task2);

stat.raweffect = avg_task1.powspctrm - avg_task2.powspctrm;
mask1 = stat.posclusterslabelmat~=0;
mask2 = stat.negclusterslabelmat~=0;
mask_plot = mask1|mask2;
stat.mask_plot = mask_plot;

% plot settings
set(0, 'DefaultAxesFontName', 'Arial');
set(0, 'DefaultAxesFontSize', 12);
set(0, 'DefaultTextFontName', 'Arial');
set(0, 'DefaultTextFontSize', 12);

dur = 0.1;
channel = {'Cz'};   
cfg = [];
cfg.parameter = 'powspctrm';
cfg.channel = channel;
cfg.xlim = [-0.5 2];
cfg.title = [""];
f = figure('units','inch','position',[0,0,10.5,4]);
cfg.figure = f;

subplot(2,3,4)
cfg.ylim = [0 50];
cfg.zlim = [-0.7 0.7];
cfg.title = "";
ft_singleplotTFR(cfg, avg_task1)
place_patches(dur, 'low')
xlabel('Time (s)') ;
ylabel('Frequency (Hz)');
box off 

subplot(2,3,5)
ft_singleplotTFR(cfg, avg_task2)
place_patches(dur, 'low')
xlabel('Time (s)') ;
ylabel('Frequency (Hz)');
box off 

cfg.parameter = 'stat';
cfg.baseline = 'no';
cfg.zlim = [-6 6];
cfg.maskparameter = 'mask_plot';
cfg.maskstyle = 'outline';
subplot(2,3,6)
ft_singleplotTFR(cfg, stat)
place_patches(dur, 'low')
xlabel('Time (s)') ;
ylabel('Frequency (Hz)');
box off 

subplot(2,3,1)
cfg.ylim = [50 100];
cfg.zlim = [-0.015 0.015];
cfg = rmfield(cfg, ["maskparameter", "maskstyle"]);
cfg.parameter = 'powspctrm';
cfg.title = task1;
ft_singleplotTFR(cfg, avg_task1)
place_patches(dur, 'high')
xlabel('Time (s)') ;
ylabel('Frequency (Hz)');
box off 

subplot(2,3,2)
cfg.title = task2;
ft_singleplotTFR(cfg, avg_task2)
place_patches(dur, 'high')
xlabel('Time (s)') ;
ylabel('Frequency (Hz)');
box off 

cfg.parameter = 'stat';
cfg.baseline = 'no';
cfg.title = 't-value';
cfg.zlim = [-6 6];
cfg.maskparameter = 'mask_plot';
cfg.maskstyle = 'outline';
subplot(2,3,3)
ft_singleplotTFR(cfg, stat)
place_patches(dur, 'high')
xlabel('Time (s)') ;
ylabel('Frequency (Hz)');
box off 

saveas(f,['/data/pt_02299/ExpSupp/Main/results/figures/' folder '/TFR.svg'])

uiwait(f)

function place_patches(dur, pos)
if strcmp(pos, 'low')
    patch('vertices', [0.5 8; 0.5 12; 0.9+dur 12; 0.9+dur 8], ...
        'faces', [1, 2, 3, 4], ...
        'FaceColor', 'none', ...
        'EdgeColor','black', ...
        'LineStyle', "--");
    patch('vertices', [0.3 13; 0.3 30; 0.6+dur 30; 0.6+dur 13], ...
        'faces', [1, 2, 3, 4], ...
        'FaceColor', 'none', ...
        'EdgeColor','black', ...
        'LineStyle', "--");
    patch('vertices', [1.5 8; 1.5 12; 1.9+dur 12; 1.9+dur 8], ...
        'faces', [1, 2, 3, 4], ...
        'FaceColor', 'none', ...
        'EdgeColor','black', ...
        'LineStyle', "--");
    patch('vertices', [1.3 13; 1.3 30; 1.6+dur 30; 1.6+dur 13], ...
        'faces', [1, 2, 3, 4], ...
        'FaceColor', 'none', ...
        'EdgeColor','black', ...
        'LineStyle', "--");
elseif strcmp(pos, 'high')
    patch('vertices', [0.15 70; 0.15 90; 0.35+dur 90; 0.35+dur 70], ...
        'faces', [1, 2, 3, 4], ...
        'FaceColor', 'none', ...
        'EdgeColor','black', ...
        'LineStyle', "--");
    patch('vertices', [1.15 70; 1.15 90; 1.35+dur 90; 1.35+dur 70], ...
        'faces', [1, 2, 3, 4], ...
        'FaceColor', 'none', ...
        'EdgeColor','black', ...
        'LineStyle', "--");
end
end
