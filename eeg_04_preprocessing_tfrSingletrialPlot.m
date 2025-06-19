%%%
% preprocessing of eeg data - single trial and electrode TFR plots
% ----------------------------------------------------------------
% runs the Time-Frequency Analysis using Fieldtrip
% Fieldtrip functions need to be added to the Matlab path
% we used Fieldtrip version 0.20220623
% the code then plots all time-frequency-representations for all trials
% and electrodes into separate figures and saves them in a pdf per subject
% do not close figure while code is running!
%
% authors:
% --------
% Lisa-Marie Pohle, Moritz Nickel
%
% contact:
% --------
% lmpohle@cbs.mpg.de
%
% date:
% -----
% 5th June 2025
%%%

subjects = {'sub-es01','sub-es02','sub-es03','sub-es04','sub-es05', ...
    'sub-es06','sub-es07','sub-es08','sub-es09','sub-es10','sub-es11', ...
    'sub-es12','sub-es13','sub-es14','sub-es15','sub-es16','sub-es17', ...
    'sub-es18','sub-es19','sub-es20','sub-es21','sub-es22','sub-es23', ...
    'sub-es24','sub-es25','sub-es26','sub-es27','sub-es28','sub-es29', ...
    'sub-es30','sub-es31','sub-es32','sub-es33','sub-es34','sub-es35', ...
    'sub-es37','sub-es38','sub-es39','sub-es40','sub-es41','sub-es42', ...
    'sub-es43','sub-es44','sub-es45','sub-es46','sub-es47'};
 
channels = {'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT9', 'FC5', ...
    'FC1', 'FCz', 'FC2', 'FC6', 'FT10', 'T7', 'C3', 'Cz', 'C4', 'T8', ...
    'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', ...
    'P8', 'O1', 'O2'};
channel_position = {5, 7, 13, 15, 17, 19, 21, 23, 25, 27, 28, 29, 31, 33, ...
    35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 71, 73};
task = ['prepForTFARejection'];
data_path = '/data/pt_02299/ExpSupp/Main/derivatives';

for isub = 1:numel(subjects)
    sub = subjects{isub};
    if ~exist(fullfile(data_path, sub, 'tfr'), 'dir')
        mkdir(fullfile(data_path, sub, 'tfr'))
    end

    tfr_CO2_single(sub, data_path, task, channels, channel_position)
end


function tfr_CO2_single(sub, data_path, task, channels, channel_position)
% load data
filein = fullfile(data_path, sub, 'eeg', [sub '_task-' task '_run-01_eeg.eeg']);

cfg = [];
cfg.dataset = filein;
cfg.channel = {'all';'-VEOG';'-HEOG';'-STI 014'};
data = ft_preprocessing(cfg);

% epoch data
trl = trialDefinition(filein);
cfg = [];
cfg.trl = trl;
data = ft_redefinetrial(cfg, data);

% time-frequency representation
cfg             = [];
cfg.method      = 'mtmconvol';
cfg.output      = 'pow';
cfg.taper       = 'hanning';
cfg.channel     = 'all';
cfg.pad         = 'maxperlen';
cfg.foi         = 1:100;
cfg.t_ftimwin   = [ones(1,nnz(cfg.foi <= 30))*0.5, ones(1,nnz(cfg.foi > 30))*0.25];
cfg.toi         = -1:0.020:3;
cfg.keeptrials  = 'yes';
tfr = ft_freqanalysis(cfg, data);

% plot single trial TFR
cfg = [];
cfg.parameter = 'powspctrm';
cfg.baseline = [-0.75, -0.25];
cfg.baselinetype   = 'relchange';
cfg.xlim = [-0.5 2];
cfg.zlim = [-1 4];
f = figure('windowState','maximized');
cfg.figure = f;

[nTrial, ~] = size(tfr.trialinfo);
for itrial = 1:nTrial
    cfg.trials = itrial;
    for ichan = 1:numel(channels)
        subplot(7,11,channel_position{ichan})
        cfg.channel = channels{ichan};
        ft_singleplotTFR(cfg,tfr)
    end
    sgtitle(['sub = ' sub ', trial = ' num2str(itrial)]);
    exportgraphics(f, fullfile(data_path, sub, 'tfr', ...
        [sub '_preparedForManualRejection_tfr.pdf']), 'Append', true);
    clf(f)
end 
end

function trl = trialDefinition(datasetName)
cfg = [];
cfg.dataset             = datasetName;
cfg.trialfun            = 'ft_trialfun_general';
cfg.trialdef.eventtype  = '1st';
cfg.trialdef.eventvalue = 1;
cfg.trialdef.prestim    = 1;
cfg.trialdef.poststim   = 3;
cfg = ft_definetrial(cfg);
trl = cfg.trl;
trl(:,4) = 1:size(trl,1);
end
