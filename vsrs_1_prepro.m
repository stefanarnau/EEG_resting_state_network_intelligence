% Clear evil residuals
clear all;

% Paths
PATH_EEGLAB  = '/home/arnau/eeglab2022.1/';
PATH_INFILES = '/mnt/storage/vital/rohdaten/DKA/01_AZ & AA/';
PATH_OUT   = '/mnt/data/arnauOn8TB/vs_resting_state/1_cleaned/';

% Init EEGlab
addpath(PATH_EEGLAB);
eeglab;

% Chanlocfile
channel_location_file = [pwd, '/standard-10-5-cap385.elp'];

% Get list of files
infiles = dir(fullfile(PATH_INFILES, '*.vhdr'));
id_strings = {};
for f = 1 : numel(infiles)
    fn = infiles(f).name;
    id_string = fn(regexp(fn, '\d'));
    id_strings{f} = id_string(1 : end - 1);
end
id_strings = unique(id_strings);

% Iterate files
failed = zeros(1, length(id_strings)); 
parfor s = 1 : length(id_strings) % Use parfor here on threadripper

    % Load raw data
    try
        DAT = {};
        DAT{1} = pop_loadbv(PATH_INFILES, ['DKA', id_strings{s}, '_AA1.vhdr'], [], [1 : 64]);
        DAT{2} = pop_loadbv(PATH_INFILES, ['DKA', id_strings{s}, '_AA2.vhdr'], [], [1 : 64]);
        DAT{3} = pop_loadbv(PATH_INFILES, ['DKA', id_strings{s}, '_AZ1.vhdr'], [], [1 : 64]);
        DAT{4} = pop_loadbv(PATH_INFILES, ['DKA', id_strings{s}, '_AZ2.vhdr'], [], [1 : 64]);

        % Code conditions
        cond_labs = {'eyes_open_1', 'eyes_open_2', 'eyes_closed_1', 'eyes_closed_2'};
        for d = 1 : numel(DAT)
            DAT{d} = eeg_regepochs(DAT{d}, 'recurrence', 4, 'extractepochs', 'off');
            for e = 1 : length(DAT{d}.event)
                if strcmpi(DAT{d}.event(e).type, 'X') & DAT{d}.event(e).latency >= 2000 % Exclude first event to prevent condition overlap
                    DAT{d}.event(e).type = cond_labs{d};
                end
            end
        end

        % Merge datasets
        EEG = pop_mergeset(DAT{1}, DAT{2});
        EEG = pop_mergeset(EEG, DAT{3});
        EEG = pop_mergeset(EEG, DAT{4});

        % Add chanlocs
        EEG = pop_chanedit(EEG, 'lookup', channel_location_file);
        EEG.chanlocs_original = EEG.chanlocs;

        % Resample data
        EEG = pop_resample(EEG, 200);

        % Bandpass filter data
        EEG = pop_basicfilter(EEG, [1 : EEG.nbchan], 'Cutoff', [1, 40], 'Design', 'butter', 'Filter', 'bandpass', 'Order', 4, 'RemoveDC', 'on', 'Boundary', 'boundary');

        % Bad channel detection / rejection
        [EEG, i1] = pop_rejchan(EEG, 'elec', [1 : EEG.nbchan], 'threshold', 5, 'norm', 'on', 'measure', 'kurt');
        [EEG, i2] = pop_rejchan(EEG, 'elec', [1 : EEG.nbchan], 'threshold', 5, 'norm', 'on', 'measure', 'prob');
        EEG.chans_rejected = horzcat(i1, i2);
        EEG.chans_rejected_n = length(horzcat(i1, i2));

        % Interpolate missing channels
        EEG = pop_interp(EEG, EEG.chanlocs_original, 'spherical');

        % Rereference data
        EEG = pop_reref(EEG, []);

        % Determine rank of data
        datarank = sum(eig(cov(double(EEG.data'))) > 1e-6);

        % Epoch dataset
        EEG = pop_epoch(EEG, cond_labs, [-4, 4], 'newname', [id_strings{s} '_epoched'], 'epochinfo', 'yes');
        EEG = pop_rmbase(EEG, []);
        EEG = eeg_checkset(EEG, 'eventconsistency');

        % Epoch rejection
        [EEG, rejsegs] = pop_autorej(EEG, 'nogui', 'on', 'threshold', 1000, 'startprob', 5, 'maxrej', 5, 'eegplot', 'off');
        EEG.n_segs_rejected = length(rejsegs);

        % Get condition indices
        EEG.cond_idx = {[], [], [], []};
        EEG.cond_labs = cond_labs;
        for e = 1 : length(EEG.event)
            [event_flag, idx] = ismember(EEG.event(e).type, cond_labs);
            if event_flag == 1
                EEG.cond_idx{idx}(end + 1) = EEG.event(e).epoch;
            end
        end
        for cond = 1 : 4
            EEG.cond_idx{cond} = unique(EEG.cond_idx{cond});
        end

        % ICA
        EEG = pop_runica(EEG, 'extended', 1, 'interrupt', 'on', 'pca', datarank);
        EEG = iclabel(EEG);

        % Remove components
        EEG.ICs_out = find(EEG.etc.ic_classification.ICLabel.classifications(:, 1) < 0.3 | EEG.etc.ic_classification.ICLabel.classifications(:, 3) > 0.3);
        EEG = pop_subcomp(EEG, EEG.ICs_out, 0);

        % Create a conditin table
        EEG.trialinfo = [];
        for e = 1 : length(EEG.epoch)

            if strcmpi(EEG.epoch(e).eventtype{1}, 'eyes_open_1')
                eyes_open = 1;
                session = 1;
            elseif strcmpi(EEG.epoch(e).eventtype{1}, 'eyes_open_2')
                eyes_open = 1;
                session = 2;
            elseif strcmpi(EEG.epoch(e).eventtype{1}, 'eyes_closed_1')
                eyes_open = 0;
                session = 1;
            elseif strcmpi(EEG.epoch(e).eventtype{1}, 'eyes_closed_2')
                eyes_open = 0;
                session = 2;
            end
            EEG.trialinfo(e, :) = [eyes_open, session];

        end

        % Save data
        [~] = pop_saveset(EEG, 'filename', [id_strings{s}, '_cleaned.set'], 'filepath', PATH_OUT);

    catch
        failed(s) = 1;
        continue;
    end
    
end % End file iteration


