%%%
% analysis of eeg data - time-frequency domain, response to first stimulus
% ------------------------------------------------------------------------
% load data
% subtractive baseline correction
% average over repetitions per participant
% calculate grand average and plot TFR
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

tfrs = [];

for isub = 1:numel(subjects)
    sub = subjects{isub};
    
    % load TFR per subject and apply baseline correction for plotting
    load(['/data/pt_02299/ExpSupp/Main/derivatives/' sub '/tfr/' sub '_task-expsupp_tfr.mat'])
    
    cfg = [];
    cfg.parameter = 'powspctrm';
    cfg.baseline = [-0.75, -0.25];
    cfg.baselinetype   = 'absolute';
    tfr_base = ft_freqbaseline(cfg, tfr);

    cfg = [];
    cfg.avgoverrpt = 'yes';
    cfg.trials = 'all';
    tfrs{isub} = ft_selectdata(cfg, tfr_base);

    disp(['################# ' sub ' loaded! #####################'])
end

cfg = [];
avg = ft_freqgrandaverage(cfg, tfrs{:});

% plot settings
set(0, 'DefaultAxesFontName', 'Arial');
set(0, 'DefaultAxesFontSize', 6);
set(0, 'DefaultTextFontName', 'Arial');
set(0, 'DefaultTextFontSize', 6);

dur = 0.1;
channel = {'Cz'};

cfg = [];
cfg.parameter = 'powspctrm';
cfg.channel = channel;
cfg.xlim = [-0.5 1];
f = figure('units','inch','position',[0,0,7/6,4/3]);
cfg.figure = f;

% create Fig2d with 2 panels
subplot(2,1,2)
cfg.ylim = [0 50];
cfg.zlim = [-0.5 0.5];
ft_singleplotTFR(cfg, avg)
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

subplot(2,1,1)
cfg.ylim = [50 100];
cfg.zlim = [-0.006 0.006];
ft_singleplotTFR(cfg, avg)
patch('vertices', [0.15 70; 0.15 90; 0.35+dur 90; 0.35+dur 70], ...
    'faces', [1, 2, 3, 4], ...
    'FaceColor', 'none', ...
    'EdgeColor','black', ...
    'LineStyle', "--");
xlabel('Time (s)') ;
ylabel('Frequency (Hz)');
box off 

sgtitle(['average, n=36, electrode Cz, Baseline: subtraction, single trial'])
saveas(f,'/data/pt_02299/ExpSupp/Main/results/figures/Fig2/TFR.svg')
uiwait(f)
