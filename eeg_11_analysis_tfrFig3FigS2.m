%%%
% analysis of eeg data - time-frequency domain, repetition suppression
% --------------------------------------------------------------------
% load data
% subtractive baseline correction
% average over repetitions per participant
% extract mean power at Cz in prespecified time-frequency bins
% save values for later analysis in JASP
% calculate and plot grand average
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

used_blocks = '14';  %enter start and end block number here, for Fig.3 its 14, for Fig.S2 18

tfrs = [];
roi_values = table('Size',[36 7], ...
    'VariableTypes',["string","double","double","double","double","double","double"], ...
    'VariableNames',["Subject", "alpha1", "alpha2", "beta1", "beta2", "gamma1", "gamma2"]);

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

    index_subset = find(strcmp(events.(['drop_log_blocks' used_blocks]), '()') & contains(events.trial_type, '1st/repetition'));

    cfg = [];
    cfg.parameter = 'powspctrm';
    cfg.baseline = [-0.75, -0.25];
    cfg.baselinetype   = 'absolute';
    tfr_base = ft_freqbaseline(cfg, tfr);

    cfg = [];
    cfg.avgoverrpt = 'yes';
    cfg.trials = index_subset;
    cfg.channel = 'Cz';
    tfrs{isub} = ft_selectdata(cfg, tfr_base);

    % get average values of power within the prespecified frequency and
    % time bins (already averaged across trials)
    roi_values(isub, 1) = {sub};
    roi_values(isub, 2) = {mean(tfrs{isub}.powspctrm(:,8:12,76:100), 'all')};
    roi_values(isub, 3) = {mean(tfrs{isub}.powspctrm(:,8:12,126:150), 'all')};
    roi_values(isub, 4) = {mean(tfrs{isub}.powspctrm(:,13:30,66:85), 'all')};
    roi_values(isub, 5) = {mean(tfrs{isub}.powspctrm(:,13:30,116:135), 'all')};
    roi_values(isub, 6) = {mean(tfrs{isub}.powspctrm(:,70:90,58:73), 'all')};
    roi_values(isub, 7) = {mean(tfrs{isub}.powspctrm(:,70:90,108:123), 'all')};

    disp(['################# ' sub ' loaded! #####################'])
end

if used_blocks == '14'
    writetable(roi_values,'/data/pt_02299/ExpSupp/Main/results/eeg/fig3_tfr_rois.csv');
elseif used_blocks == '18'
    writetable(roi_values,'/data/pt_02299/ExpSupp/Main/results/eeg/figS2_tfr_rois.csv');
else
    error('Saving of other blocks than 1-4 or 1-8 currently not supported.')
end

cfg = [];
avg_tfr = ft_freqgrandaverage(cfg, tfrs{:});

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
f = figure('units','inch','position',[0,0,3.5,4]);
cfg.figure = f;

% create Fig3c with 2 panels
subplot(2,1,2)
cfg.ylim = [0 50];
cfg.zlim = [-0.7 0.7];
ft_singleplotTFR(cfg, avg_tfr)
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
xlabel('Time (s)') ;
ylabel('Frequency (Hz)');
box off 

subplot(2,1,1)
cfg.ylim = [50 100];
cfg.zlim = [-0.015 0.015];
ft_singleplotTFR(cfg, avg_tfr)
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
xlabel('Time (s)') ;
ylabel('Frequency (Hz)');
box off 

sgtitle(['Repetition trials, n=36, electrode Cz'])

if used_blocks == '14'
    saveas(f,'/data/pt_02299/ExpSupp/Main/results/figures/Fig3/TFR.svg')
elseif used_blocks == '18'
    saveas(f,'/data/pt_02299/ExpSupp/Main/results/figures/FigS2/TFR.svg')
else
    error('Saving of other blocks than 1-4 or 1-8 currently not supported.')
end

uiwait(f)
