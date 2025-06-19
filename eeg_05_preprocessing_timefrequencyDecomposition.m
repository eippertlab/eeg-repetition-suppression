%%%
% preprocessing of eeg data - time-frequency decomposition
% --------------------------------------------------------
% runs the Time-Frequency Analysis using Fieldtrip
% Fieldtrip functions need to be added to the Matlab path
% we used Fieldtrip version 0.20220623
%
% authors:
% --------
% Moritz Nickel, Lisa-Marie Pohle
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

for isub = 1:numel(subjects)
    sub = subjects{isub}
    if ~exist(['/data/pt_02299/ExpSupp/Main/derivatives/' sub '/tfr/'], 'dir')
        mkdir(['/data/pt_02299/ExpSupp/Main/derivatives/' sub '/tfr/'])
    end

    tfr_CO2_single(sub)
end

function tfr_CO2_single(sub)
filein = ['/data/pt_02299/ExpSupp/Main/derivatives/' sub '/eeg/' sub '_task-expsupp_eeg.eeg'];

cfg = [];
cfg.dataset = filein;
cfg.channel = {'all';'-VEOG';'-HEOG';'-STI 014'};
data = ft_preprocessing(cfg);

% epoch data
trl = trialDefinition(filein);
cfg = [];
cfg.trl = trl;
dataTrials = ft_redefinetrial(cfg, data);

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
tfr = ft_freqanalysis(cfg, dataTrials);

save(['/data/pt_02299/ExpSupp/Main/derivatives/' sub '/tfr/' sub '_task-expsupp_tfr.mat'], 'tfr')
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