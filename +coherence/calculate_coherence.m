function [coherence, inputdata, cov, pooled_coherence, comparison_of_coherence, pooled_participant_coherence] = calculate_coherence()

%% CALCULATE_COHERENCE - Main function to analyse EMG and force data for intermuscular coherence
%
% This script analyses EMG and force data from dexterity- and strength-trained groups 
% to calculate intermuscular coherence using the NeuroSpec20 toolbox by David Halliday et al.  
% between abductor pollicis brevis and flexor digitorum superficialis muscles. 
% 
% The analysis follows these main steps:
%   1. Load and preprocess EMG and force data
%   2. Identify the period of minimum coefficient of variation of force for
%      each submaximal force level.
%   3. Calculate individual coherence for each participant during this
%      period.
%   4. Pool coherence within participants across trials.
%   5. Pool coherence across participants within groups.
%   6. Compare coherence between groups.
%
% Returns:
%   coherence - structure containing individual coherence results for each participant
%   inputdata - structure containing EMG and force data used in analysis
%   cov - structure containing coefficient of variation results for force data
%   pooled_coherence - structure containing pooled coherence results within groups
%   comparison_of_coherence - structure containing statistical comparison between groups
%   pooled_participant_coherence - structure containing coherence pooled within participants across trials
%
% Usage:
%   [coherence_data, inputdata, cov, pooled_coherence, comparison_of_coherence, pooled_participant_coherence] = coherence.calculate_coherence();
%
% Notes:
%   Requires the NeuroSpec20 toolbox by David Halliday.
%   This can be downloaded from: https://github.com/dmhalliday/NeuroSpec/blob/master/neurospec20.zip 
%   Once downloaded and unzipped, place the entire folder within the folder "+coherence"
%   ALL RIGHTS TO DAVID HALLIDAY. David Halliday et al. have allowed the use of this toolbox under a GNU general public license.



warning('off', 'MATLAB:table:ModifiedAndSavedVarnames'); % Turns off unnecessary warning

% Add path to the NeuroSpec Toolbox
addpath(fullfile(pwd, '+coherence', 'neurospec20', 'neurospec20'));

% Set up analysis parameters and initialise data structures
CONFIG = initialise_config();
GROUPS = define_groups();
[coherence, inputdata, cov] = initialise_output_structures();

% Run main analysis pipeline
[coherence, inputdata, cov] = process_all_participants(GROUPS, CONFIG, coherence, inputdata, cov);
[coherence, pooled_participant_coherence] = pool_participant_coherence(coherence, CONFIG);
[pooled_coherence, comparison_of_coherence] = calculate_pooled_coherence(coherence, CONFIG);
end


%% CONFIGURATION FUNCTIONS:

function CONFIG = initialise_config()
% initialise analysis configuration parameters
%
% Returns a structure containing all configuration parameters:
%   FORCE_LEVELS - Force levels as percentage of maximum voluntary contraction
%   REPS - Number of repetitions per force level
%   FS - Sampling frequency in Hz
%   WINDOW_DURATION - Duration of analysis window in milliseconds
%   STEP_SIZE - Step size for sliding window analysis
%   SAMPLING_RATE - Data acquisition rate
%   DATA_PATH - Path to data directory

CONFIG = struct();
CONFIG.FORCE_LEVELS = {'15', '35', '55', '70'};  % Force levels as % of MVC
CONFIG.REPS = {'1', '2', '3'};                   % Number of repetitions
CONFIG.FS = 2222;                                % Sampling frequency (Hz)
CONFIG.WINDOW_DURATION = 4000;                   % Window duration (ms)
CONFIG.STEP_SIZE = 50;                          % Step size for sliding window
CONFIG.SAMPLING_RATE = 2000;                    % Force data sampling rate
CONFIG.DATA_PATH = fullfile(pwd, '+coherence', '+emg_force_data', filesep);
end

function GROUPS = define_groups()
% Define participant groups for analysis
%
% Returns structure with two fields:
%   strength - Array of participant IDs for climber group
%   dexterity - Array of participant IDs for musician group

GROUPS = struct();
GROUPS.strength = [3, 4, 5, 6, 7, 8, 9, 11, 16, 20];
GROUPS.dexterity = [1, 2, 10, 12, 13, 14, 17, 18, 19, 21];
end	



%% DATA PROCESSING FUNCTIONS

function [coherence, inputdata, cov] = initialise_output_structures()
% initialise empty structures for storing analysis results
%
% Returns:
%   coherence - Structure for coherence results
%   inputdata - Structure for processed EMG/force data
%   cov - Structure for coefficient of variation results

coherence = struct('dexterity', struct(), 'strength', struct());
inputdata = struct('dexterity', struct(), 'strength', struct());
cov = struct('dexterity', struct(), 'strength', struct());
end

function [coherence, inputdata, cov] = process_all_participants(GROUPS, CONFIG, coherence, inputdata, cov)
% Process data for all participants in each group
%
% For each participant:
%   1. Loads EMG and force data
%   2. Calculates force metrics
%   3. Identifies period of smallest cov of force (steadiest period)
%   4. Calculates coherence
%
% Parameters:
%   GROUPS - Structure defining participant groups
%   CONFIG - Analysis configuration parameters
%   coherence, inputdata, cov - Empty structures to store results
%
% Returns updated versions of input structures with processed data

group_names = fieldnames(GROUPS);

for group_idx = 1:length(group_names)
    group_name = group_names{group_idx};
    participants = GROUPS.(group_name);

    for participant = participants
        try
            % Process individual participant data
            [participant_data, participant_coherence, participant_cov] = process_participant(participant, CONFIG);

            % Store results in output structures
            participant_field = sprintf('P%d', participant);
            coherence.(group_name).(participant_field) = participant_coherence;
            inputdata.(group_name).(participant_field) = participant_data;
            cov.(group_name).(participant_field) = participant_cov;
        catch ME
            warning('Failed to process participant %d: %s', participant, ME.message);
            continue;
        end
    end
end
end

function [data, coherence_out, cov_out] = process_participant(participant, config)
% Process all data for a single participant
%
% Workflow:
%   1. Load EMG data
%   2. Load force data
%   3. Calculate max force
%   4. Find period of steadiest force (minimum cov)
%   5. Calculate intermuscular coherence during this period
%
% Parameters:
%   participant - Participant ID number
%   config - Analysis configuration parameters
%
% Returns:
%   data - Structure containing processed EMG and force data
%   coherence_out - Structure containing coherence results
%   cov_out - Structure containing force variability metrics

% Load and process EMG data
emg_data = load_emg_data(participant, config);

% Load and process force data
[force_data, max_force] = load_force_data(participant, config);

% Calculate force metrics
[force_cov_target] = calculate_force_metrics(force_data, max_force);

% Find periods of steady force
periods = find_steady_periods(force_data, config);

% Calculate coherence metrics
fprintf('\nProcessing P%d Coherence:\n', participant)
coherence_results = calculate_coherence_metrics(emg_data, periods, config);

% Package outputs
data = struct('emg', emg_data, 'force', force_data);
coherence_out = coherence_results;
cov_out = struct('forceCOVtarget', force_cov_target);
end

function emg_data = load_emg_data(participant, config)
% Load and preprocess EMG data for all trials
%
% For each force level and repetition:
%   1. Loads CSV file
%   2. Applies high-pass butterworth filter at 20Hz
%   3. Organises data into 4D array
%
% Parameters:
%   participant - Participant ID number
%   config - Configuration parameters
%
% Returns:
%   emg_data - 4D array [time x channels x repetitions x force_levels]

emg_data = zeros(45551, 9, length(config.REPS), length(config.FORCE_LEVELS));

for i = 1:length(config.FORCE_LEVELS)
    for j = 1:length(config.REPS)
        % Construct filename for current trial
        filename = fullfile(config.DATA_PATH, ...
            sprintf('P%d_%sMVC_Rep_%s.csv', participant, config.FORCE_LEVELS{i}, config.REPS{j}));

        if ~isfile(filename)
            error('EMG file not found: %s', filename);
        end

        % Load and preprocess EMG data
        emg = readtable(filename);
        emg = table2array(emg);
        [B, A] = butter(2, 20 / (config.FS / 2), 'high'); % set parameters for high-pass 2nd order butterworth filter at 20Hz
        emg = [emg(:,1), filtfilt(B, A, emg(:,2:9))]; % run Butterworth filter
        emg = [emg(:,1), zscore(emg(:,2:9))]; % Z-normalise channels
        emg = [emg(:,1), abs(hilbert(emg(:,2:9)))]; % Rectify via the absolute value of the signals Hilbert transform
        emg_data(:,:,j,i) = emg;
    end
end
end

function [force_data, max_force] = load_force_data(participant, config)
% Load and process force data
%
% Steps:
%   1. Load maximum voluntary contraction (MVC) data
%   2. Load submaximal force trials
%   3. Split trials into three sections
%
% Parameters:
%   participant - Participant ID number
%   config - Configuration parameters
%
% Returns:
%   force_data - 4D array [time x channels x repetitions x force_levels]
%   max_force - Maximum force from MVC trial

% Load MVC force data
mvc_filename = fullfile(config.DATA_PATH, sprintf('P%d_MVC_Force.txt', participant));
if ~isfile(mvc_filename)
    error('MVC force file not found: %s', mvc_filename);
end

mvc_force = readtable(mvc_filename);
mvc_force = table2array(mvc_force);
mvc_force = mvc_force(~any(isnan(mvc_force), 2), :);
max_force = max(mvc_force(:, 2));

% Load submaximal force data
force_data = zeros(40000, 2, 3, length(config.FORCE_LEVELS));
for i = 1:length(config.FORCE_LEVELS)
    filename = fullfile(config.DATA_PATH, ...
        sprintf('P%d_%sMVC_Force.txt', participant, config.FORCE_LEVELS{i}));

    if ~isfile(filename)
        error('Force file not found: %s', filename);
    end

    force = readtable(filename);
    force = table2array(force);

    % Split force data into three sections
    force_data(:,:,1,i) = force(3:40002, 1:2); % This is rep 1
    force_data(:,:,2,i) = force(40012:80011, 1:2); % This is rep 2
    force_data(:,:,3,i) = force(80021:120020, 1:2); % This is rep 3
end
end

function [force_cov_target] = calculate_force_metrics(force_data, max_force)
% Calculate force coefficient of variation in relation to the target force
% for center 6 s of force data during steady period.
%
% Parameters:
%   force_data - Force measurements
%   max_force - Maximum force from MVC
%
% Returns:
%   force_cov_target

steady_force = force_data(14000:26000, 1:2, :, :); % Clip data to 6 seconds during force steady period

% Calculate target-based COV
target = [0.15, 0.35, 0.55, 0.70]' * max_force;
force_cov_target = zeros(3, 4);
for i = 1:4
    for j = 1:3
        force_cov_target(j,i) = std(steady_force(:,2,j,i)) / target(i);
    end
end
force_cov_target = mean(force_cov_target, 1);
end

function periods = find_steady_periods(force_data, config)
% Identify periods of steady force using sliding window analysis
%
% Uses sliding window to calculate COV and finds period with minimum variability
%
% Parameters:
%   force_data - Force measurements
%   config - Configuration parameters
%
% Returns:
%   periods - Array of steady force periods [start_time end_time COV]

periods = zeros(1, 3, 3, 4);

for i = 1:4
    for j = 1:3
        % Calculate CV using sliding window
        results = sliding_window_cov(force_data(:,2,j,i), ...
            force_data(1,2,j,i), ...
            config.WINDOW_DURATION, ...
            config.STEP_SIZE, ...
            config.SAMPLING_RATE);

        % Find period with minimum CV within valid time range
        valid_periods = results(results(:,1) >= 7 & results(:,2) <= 13, :);
        if ~isempty(valid_periods)
            [~, min_idx] = min(valid_periods(:,3));
            periods(1,:,j,i) = valid_periods(min_idx,:);
        else
            periods(1,:,j,i) = [NaN, NaN, NaN];
        end
    end
end
end

function coherence_results = calculate_coherence_metrics(emg_data, periods, config)
% Calculate coherence metrics for EMG data
%
% Uses the neurospec20 toolbox's sp2a2_m1 function to calculate coherence
% metrics.
%
% Parameters:
%   emg_data - EMG measurements
%   periods - Steady force periods
%   config - Configuration parameters
%
% Returns:
%   coherence_results - Structure with coherence metrics

% Preallocate variables output from sp2a2_m1
f = zeros(512, 5, 3, 4);
t = zeros(1024, 2, 3, 4);
sc = complex(zeros(513, 3, 3, 4));
cl = struct('type', [], 'seg_size', [], 'seg_tot', [], 'seg_tot_var', [], ...
           'samp_tot', [], 'samp_rate', [], 'dt', [], 'df', [], ...
           'f_c95', [], 'ch_c95', [], 'q_c95', [], ...
           'N1', [], 'N2', [], 'P1', [], 'P2', [], ...
           'opt_str', char(zeros(1,0)), 'what', char(zeros(1,0)));
cl = repmat(cl, [1 1 3 4]);

for i = 1:4
    for j = 1:3
        % Extract steady state EMG data
        start_time = round(periods(:,1,j,i) * config.FS);
        end_time = round(periods(:,2,j,i) * config.FS);
        steady_emg = emg_data(start_time:end_time,:,j,i);

        % Calculate coherence using sp2a2_m1 function
        [f(:,:,j,i), t(:,:,j,i), cl(:,:,j,i), sc(:,:,j,i)] = ...
            sp2a2_m1(0, steady_emg(:,2), steady_emg(:,6), config.FS, 10);  
    end
end

% Package results
coherence_results = struct('f', f, 't', t, 'cl', cl, 'sc', sc, ...
    'period_smallest_cov', periods);
end

function [coherence_out, pooled_participant] = pool_participant_coherence(coherence, config)
% Pool coherence data within each participant across trials
%
% For each participant:
%   1. Pools coherence across repetitions for each force level
%   2. Uses pool_scf for combining coherence estimates
%   3. Uses pool_scf_out to generate ouput pooled results for group pooling
%
% Parameters:
%   coherence - Structure containing individual coherence results
%   config - Analysis configuration parameters
%
% Returns:
%   coherence_out - Updated coherence structure
%   pooled_participant - Structure containing pooled coherence per participant

pooled_participant = struct('dexterity', struct(), 'strength', struct());
coherence_out = coherence;

% Setup group and force level labels
group_names = {'dexterity', 'strength'};
force_level_labels = cellfun(@(x) sprintf('MVC%s', x), config.FORCE_LEVELS, 'UniformOutput', false);

% Loop through each group
for groupIdx = 1:length(group_names)
    group_name = group_names{groupIdx};
    participants = fieldnames(coherence.(group_name));

    % Process each participant
    for participant_loop = 1:length(participants)
        current_participant = participants{participant_loop};

        % Pool coherence across repetitions for each force level
        for i = 1:length(config.FORCE_LEVELS)
            fprintf('\nPooling %s Coherence for %sMVC:\n', current_participant, config.FORCE_LEVELS{i});

            % Pool across repetitions using pool_scf
            for j = 1:length(config.REPS)
                if j == 1
                    % initialise pooling for first repetition
                    [out_f, out_v] = pool_scf(coherence.(group_name).(current_participant).sc(:,:,j,i), ...
                        coherence.(group_name).(current_participant).cl(:,:,j,i));
                else
                    % Add subsequent repetitions to pool
                    [out_f, out_v] = pool_scf(coherence.(group_name).(current_participant).sc(:,:,j,i), ...
                        coherence.(group_name).(current_participant).cl(:,:,j,i), in_f, in_v);
                end
                in_f = out_f;
                in_v = out_v;
            end
            
            % Generate final pooled results for this force level
            [f, t, cl, sc] = pool_scf_out(in_f, in_v);
            pooled_participant.(group_name).(current_participant).(force_level_labels{i}) = ...
                struct('f', f, 't', t, 'cl', cl, 'sc', sc);
        end
    end
end
end

function [pooled_coherence, comparison] = calculate_pooled_coherence(coherence, config)
% Calculate pooled coherence across participants and compare between groups
%
% This updated version directly pools the original coherence data without requiring
% the intermediate pooled_participant_coherence step. It loops through each participant 
% and each trial to pool all data together within each group and force level.
%
% Parameters:
%   coherence - Structure containing individual coherence results for each participant
%   config - Analysis configuration parameters
%
% Returns:
%   pooled_coherence - Coherence pooled across participants within groups
%   comparison - Statistical comparison between groups

% Initialize output structures
pooled_coherence = struct('dexterity', struct(), 'strength', struct());
comparison = struct();

% Setup group and force level labels
group_names = {'dexterity', 'strength'};
force_level_labels = cellfun(@(x) sprintf('MVC%s', x), config.FORCE_LEVELS, 'UniformOutput', false);

% Pool coherence across participants within each group
for groupIdx = 1:length(group_names)
    group_name = group_names{groupIdx};
    participants = fieldnames(coherence.(group_name));

    % Process each force level
    for forceIdx = 1:length(config.FORCE_LEVELS)
        current_force_level = force_level_labels{forceIdx};
        
        % Print current processing information
        if groupIdx == 1
            fprintf('\nPooling Coherence across Participants for Dexterity-Trained at %sMVC:\n', config.FORCE_LEVELS{forceIdx});
        else
            fprintf('\nPooling Coherence across Participants for Strength-Trained at %sMVC:\n', config.FORCE_LEVELS{forceIdx});
        end
        
        % Initialize pooling variables
        in_f = [];
        in_v = [];
        
        % Pool across participants and trials
        for participantIdx = 1:length(participants)
            current_participant = participants{participantIdx};
            
            % Loop through each repetition/trial for this participant
            for trialIdx = 1:length(config.REPS)
                % Pool data - handle first entry separately
                if participantIdx == 1 && trialIdx == 1
                    [out_f, out_v] = pool_scf(coherence.(group_name).(current_participant).sc(:,:,trialIdx,forceIdx), ...
                                            coherence.(group_name).(current_participant).cl(:,:,trialIdx,forceIdx));
                else
                    [out_f, out_v] = pool_scf(coherence.(group_name).(current_participant).sc(:,:,trialIdx,forceIdx), ...
                                            coherence.(group_name).(current_participant).cl(:,:,trialIdx,forceIdx), ...
                                            in_f, in_v);
                end
                
                % Update pooling variables for next iteration
                in_f = out_f;
                in_v = out_v;
            end
        end
        
        % Generate final pooled results for this force level and group
        [f, t, cl, sc] = pool_scf_out(in_f, in_v);
        pooled_coherence.(group_name).(current_force_level) = struct('f', f, 't', t, 'cl', cl, 'sc', sc);
    end
end

% Compare coherence between groups for each force level
for i = 1:length(force_level_labels)
    current_force_level = force_level_labels{i};
    
    % Use sp2_compcoh to compare strength vs dexterity
    fprintf('\nComparison of Coherence at %sMVC:\n', config.FORCE_LEVELS{i});

    [f, cl] = sp2_compcoh(...
        pooled_coherence.strength.(current_force_level).sc, ...
        pooled_coherence.strength.(current_force_level).cl, ...
        pooled_coherence.dexterity.(current_force_level).sc, ...
        pooled_coherence.dexterity.(current_force_level).cl);
    comparison.(current_force_level) = struct('f', f, 'cl', cl);
end
end

% ADDITIONAL HELPER FUNCTIONS %

% Function for calculating the period of minimal cov using a 
% sliding window analysis
function results = sliding_window_cov(force_data, target_force, window_duration, step_size, sampling_rate)

    % Convert time parameters to samples
    window_samples = window_duration * sampling_rate / 1000;
    step_samples = step_size * sampling_rate / 1000;
    
    % Calculate number of windows
    n_windows = floor((length(force_data) - window_samples) / step_samples) + 1;
    
    % Preallocate results array
    results = zeros(n_windows, 3);
    
    % Sliding window calculation
    for i = 1:n_windows
        start_idx = (i-1) * step_samples + 1;
        end_idx = start_idx + window_samples - 1;
        
        window_data = force_data(start_idx:end_idx);
        std_force = std(window_data);
        cv_value = std_force / target_force;
        
        start_time = (start_idx - 1) / sampling_rate;
        end_time = (end_idx - 1) / sampling_rate;
        
        results(i, :) = [start_time, end_time, cv_value];
    end
end