%%%
% habituation of eeg data - time-frequency domain
% -----------------------------------------------
% load data and split up runs
% subtractive baseline correction
% average over repetitions per participant
% extract mean power at Cz in prespecified time-frequency bins
% save values for later analysis in JASP
% perform cluster-based permutation test between first and second half of experiment
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


layout_file  = '/afs/cbs.mpg.de/software/fieldtrip/0.20220623/debian-bullseye-amd64/source/template/layout/easycapM24.mat';
elec = ft_read_sens('/afs/cbs.mpg.de/software/fieldtrip/0.20220623/debian-bullseye-amd64/source/template/electrode/easycap-M1.txt');

subset14 = cell(1,numel(subjects));
subset58 = cell(1,numel(subjects));
roi_values = table('Size',[36 7], ...
    'VariableTypes',["string","double","double","double","double","double", ...
    "double"],'VariableNames',["Subject", "alpha_14", "alpha_58", ...
    "beta_14", "beta_58", "gamma_14", "gamma_58"]);

for isub = 1:numel(subjects)
    sub = subjects{isub};

    load(['/data/pt_02299/ExpSupp/Main/derivatives/' sub '/tfr/' sub '_task-expsupp_tfr.mat'])
    events = readtable(['/data/pt_02299/ExpSupp/Main/derivatives/' sub '/eeg/' sub '_woAborts_hpfilter_manReject_muscleCleaned_ICsRejected_autoReject50_manReject_equalizedCounts_events.tsv'], "FileType","delimitedtext");
    index_toDelete = find(strcmp(events.drop_log_blocks18, '(''MAN REJECT POST ICA'',)'));
    index_toDelete = cat(1, index_toDelete, find(strcmp(events.drop_log_blocks18, '(''AUTO REJECT POST ICA'',)')));
    index_toDelete = cat(1, index_toDelete, find(strcmp(events.drop_log_blocks18, '(''MAN REJECT PRE ICA'',)')));
    index_toDelete = cat(1, index_toDelete, find(strcmp(events.drop_log_blocks18, '(''USER'',)')));
    index_toDelete = cat(1, index_toDelete, find(strcmp(events.drop_log_blocks18, '(''INVALID TRIAL'',)')));
    index_toDelete = cat(1, index_toDelete, find(strcmp(events.drop_log_blocks18, '(''LASER ABORT'',)')));
    events(index_toDelete,:) = [];

    index_subset14 = find(events.Run <= 4);
    index_subset58 = find(events.Run >= 5);

    cfg = [];
    cfg.avgoverrpt = 'no';
    cfg.trials = index_subset14;
    subset14_epochs = ft_selectdata(cfg, tfr);
    cfg.trials = index_subset58;
    subset58_epochs = ft_selectdata(cfg, tfr);

    cfg = [];
    cfg.parameter = 'powspctrm';
    cfg.baseline = [-0.75, -0.25];
    cfg.baselinetype   = 'absolute';
    subset14_epochs_base = ft_freqbaseline(cfg, subset14_epochs);
    subset58_epochs_base = ft_freqbaseline(cfg, subset58_epochs);

    cfg = [];
    cfg.avgoverrpt = 'yes';
    cfg.trials = 'all';
    subset14{isub} = ft_selectdata(cfg, subset14_epochs_base);
    subset58{isub} = ft_selectdata(cfg, subset58_epochs_base);

    cfg = [];
    cfg.avgoverrpt = 'yes';
    cfg.trials = 'all';
    cfg.channel = 'Cz';
    subset_for_roi_14 = ft_selectdata(cfg, subset14_epochs_base);
    subset_for_roi_58 = ft_selectdata(cfg, subset58_epochs_base);

    % get average values of power within the prespecified frequency and
    % time bins (already averaged across trials)
    roi_values(isub, 1) = {sub};
    roi_values(isub, 2) = {mean(mean(subset_for_roi_14.powspctrm(:,8:12,76:100)))};
    roi_values(isub, 4) = {mean(mean(subset_for_roi_14.powspctrm(:,13:30,66:85)))};
    roi_values(isub, 6) = {mean(mean(subset_for_roi_14.powspctrm(:,70:90,58:73)))};
    roi_values(isub, 3) = {mean(mean(subset_for_roi_58.powspctrm(:,8:12,76:100)))};
    roi_values(isub, 5) = {mean(mean(subset_for_roi_58.powspctrm(:,13:30,66:85)))};
    roi_values(isub, 7) = {mean(mean(subset_for_roi_58.powspctrm(:,70:90,58:73)))};

    subset14{isub}.elec = elec;
    subset58{isub}.elec = elec;

    disp(['################# ' sub ' loaded! #####################'])
end

writetable(roi_values,'/data/pt_02299/ExpSupp/Main/results/eeg/tfa_habituation.csv')

% prepare clusterbased permutation test
cfg = [];
cfg.keepindividual = 'yes';
grandavg14 = ft_freqgrandaverage(cfg, subset14{:});
grandavg58 = ft_freqgrandaverage(cfg, subset58{:});
grandavg14.elec = elec;

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
cfg.numrandomization = 1000;
% prepare_neighbours determines what sensors may form clusters
cfg_neighb.method    = 'distance';
cfg_neighb.channel = 'all';
cfg_neighb.neighbourdist = 60;
cfg.neighbours       = ft_prepare_neighbours(cfg_neighb, grandavg14);

design = [ones(1,size(grandavg14.powspctrm,1)), ones(1,size(grandavg14.powspctrm,1))*2;
    1:size(grandavg14.powspctrm,1), 1:size(grandavg14.powspctrm,1)];

cfg.design           = design;
cfg.ivar             = 1;
cfg.uvar             = 2;

[stat14] = ft_freqstatistics(cfg, grandavg14, grandavg58);

cfg = [];
avg14 = ft_freqdescriptives(cfg, grandavg14);
avg58 = ft_freqdescriptives(cfg, grandavg58);

stat14.raweffect = avg58.powspctrm - avg14.powspctrm;
mask1 = stat14.posclusterslabelmat~=0;
mask2 = stat14.negclusterslabelmat~=0;
mask_plot = mask1|mask2;
stat14.mask_plot = mask_plot;

dur = 0.1;
channel = {'Cz'};   
cfg = [];
cfg.parameter = 'powspctrm';
cfg.channel = channel;
cfg.xlim = [-0.5 2]; %[-0.5 1];
f = figure('WindowState','maximized');
cfg.figure = f;

subplot(2,3,4)
cfg.ylim = [0 50];
cfg.zlim = [-0.7 0.7];
cfg.title = "";
ft_singleplotTFR(cfg, avg58)
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
xlabel('Time (s)') ;
ylabel('Frequency (Hz)');
box off 

subplot(2,3,5)
ft_singleplotTFR(cfg, avg14)
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
xlabel('Time (s)') ;
ylabel('Frequency (Hz)');
box off 

cfg.parameter = 'stat';
cfg.baseline = 'no';
cfg.zlim = [-6 6];
cfg.maskparameter = 'mask_plot';
cfg.maskstyle = 'outline';
subplot(2,3,6)
ft_singleplotTFR(cfg, stat14)
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
xlabel('Time (s)') ;
ylabel('Frequency (Hz)');
box off 

subplot(2,3,1)
cfg.ylim = [50 100];
cfg.zlim = [-0.015 0.015];
cfg = rmfield(cfg, ["maskparameter", "maskstyle"]);
cfg.parameter = 'powspctrm';
cfg.title = 'Blocks 5-8';
ft_singleplotTFR(cfg, avg58)
patch('vertices', [0.15 70; 0.15 90; 0.35+dur 90; 0.35+dur 70], ...
    'faces', [1, 2, 3, 4], ...
    'FaceColor', 'none', ...
    'EdgeColor','black', ...
    'LineStyle', "--");
xlabel('Time (s)') ;
ylabel('Frequency (Hz)');
box off 

subplot(2,3,2)
cfg.title = 'Blocks 1-4';
ft_singleplotTFR(cfg, avg14)
patch('vertices', [0.15 70; 0.15 90; 0.35+dur 90; 0.35+dur 70], ...
    'faces', [1, 2, 3, 4], ...
    'FaceColor', 'none', ...
    'EdgeColor','black', ...
    'LineStyle', "--");
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
ft_singleplotTFR(cfg, stat14)
patch('vertices', [0.15 70; 0.15 90; 0.35+dur 90; 0.35+dur 70], ...
    'faces', [1, 2, 3, 4], ...
    'FaceColor', 'none', ...
    'EdgeColor','black', ...
    'LineStyle', "--");
xlabel('Time (s)') ;
ylabel('Frequency (Hz)');
box off 

sgtitle(channel)

