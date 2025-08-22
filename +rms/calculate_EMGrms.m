%% RMS analysis to identify differences in muscle activation strategy

%% Calculate RMS values across each EMG trial

% Load data 
%  - saved output from calculate_coherence analysis 
%           * Force data 
%           * EMG data (20Hz high-pass filtered, z-normalised and rectified)
load("+rms/rms_inputdata.mat")

% Set parameters
groups = {'strength','dexterity'};
window_length_ms = 500; % Window length in milliseconds
fs_emg = 2222; % EMG sampling frequency in Hz
fs_force = 2000; % Force sampling frequency in Hz

window_size_emg = round((window_length_ms/1000) * fs_emg); % EMG window size (≈1111 samples)

% Initialize the rms_data struct with the same structure as input_data
rms_data = struct();

% Loop through groups
for groupsi = 1:2
    group = groups{groupsi};
    participants = fieldnames(inputdata.(group));
    
    % Loop through participants
    for participantsi = 1:length(participants)
        participant = participants{participantsi};

        % Initialize the participant in rms_data if not exists
        if ~isfield(rms_data, group)
            rms_data.(group) = struct();
        end

        if ~isfield(rms_data.(group), participant)
            rms_data.(group).(participant) = struct();
        end

        % Loop through trials and force levels
        for triali = 1:3
            for forceleveli = 1:4
                
                % Calculate the EMG signal length
                emg_signal_length = size(inputdata.(group).(participant).emg(:,1,triali,forceleveli), 1);

                % Get the force signal length
                force_signal_length = size(inputdata.(group).(participant).force(:,2,triali,forceleveli), 1);

                % Calculate number of non-overlapping windows for EMG
                num_windows = floor(emg_signal_length / window_size_emg);

                % Get the desired EMG signals (channel 2 and 6)
                time_data = inputdata.(group).(participant).emg(:,1,triali,forceleveli);
                emg_ch2 = inputdata.(group).(participant).emg(:,2,triali,forceleveli);
                emg_ch6 = inputdata.(group).(participant).emg(:,6,triali,forceleveli);

                % Get the force data
                force = inputdata.(group).(participant).force(:,2,triali,forceleveli);

                % Get force time data
                force_time = inputdata.(group).(participant).force(:,1,triali,forceleveli);


                % Initialize arrays for RMS values and average force
                rms_ch2_values = zeros(num_windows, 1);
                rms_ch6_values = zeros(num_windows, 1);
                avg_force_values = zeros(num_windows, 1);
                time_points = zeros(num_windows, 1);

                % Calculate RMS for each non-overlapping window
                for window_idx = 1:num_windows
                    % Calculate window indices for EMG
                    emg_start_idx = (window_idx-1) * window_size_emg + 1;
                    emg_end_idx = min(window_idx * window_size_emg, emg_signal_length);

                    % Calculate the time window boundaries
                    window_start_time = time_data(emg_start_idx);
                    window_end_time = time_data(emg_end_idx);

                    % Extract window data for EMG channels
                    window_ch2 = emg_ch2(emg_start_idx:emg_end_idx);
                    window_ch6 = emg_ch6(emg_start_idx:emg_end_idx);

                    % Find force data points that fall within this time window
                    force_indices = find(force_time >= window_start_time & force_time <= window_end_time);

                    % Calculate RMS for EMG channels
                    rms_ch2_values(window_idx) = rms(window_ch2);
                    rms_ch6_values(window_idx) = rms(window_ch6);

                    % Calculate average force if there are valid force indices
                    if ~isempty(force_indices)
                        avg_force_values(window_idx) = mean(force(force_indices));
                    else
                        avg_force_values(window_idx) = NaN;
                    end

                    % Calculate time point for this window (center of window in seconds)
                    time_points(window_idx) = (window_start_time + window_end_time) / 2;
                end

                % Store the data in the rms_data struct
                if ~isfield(rms_data.(group).(participant), 'time')
                    rms_data.(group).(participant).time = {};
                end
                if ~isfield(rms_data.(group).(participant), 'force')
                    rms_data.(group).(participant).force = {};
                end
                if ~isfield(rms_data.(group).(participant), 'fds')
                    rms_data.(group).(participant).fds = {};
                end
                if ~isfield(rms_data.(group).(participant), 'apb')
                    rms_data.(group).(participant).apb = {};
                end

                % Store data in cell arrays
                rms_data.(group).(participant).time{triali, forceleveli} = time_points;
                rms_data.(group).(participant).force{triali, forceleveli} = avg_force_values;
                rms_data.(group).(participant).fds{triali, forceleveli} = rms_ch2_values;
                rms_data.(group).(participant).apb{triali, forceleveli} = rms_ch6_values;
            end
        end
    end
end

clearvars -except inputdata rms_data

%% Prepare data for export to R

% Initialize arrays for storing data
dataCell = {};
forceLevels = [15, 35, 55, 70];
group_names = {'strength', 'dexterity'};

% Process both groups
for g = 1:length(group_names)
    groupName = group_names{g};
    participants = fieldnames(rms_data.(groupName));
    
    for p = 1:length(participants)
        participantID = participants{p};
        
        % Extract numeric ID
        numericID = str2double(regexprep(participantID, '^P', ''));
        
        % Process each trial and force level
        for trials = 1:3
            for force_levels = 1:4
                % Extract first 30 time points (15s of data)
                timeVec = rms_data.(groupName).(participantID).time{trials, force_levels}(1:30);
                apbVec = rms_data.(groupName).(participantID).apb{trials, force_levels}(1:30);
                fdsVec = rms_data.(groupName).(participantID).fds{trials, force_levels}(1:30);
                forceVec = rms_data.(groupName).(participantID).force{trials, force_levels}(1:30);
                
                % Create data block for this combination
                nPoints = length(timeVec);
                dataBlock = {
                    repmat({groupName}, nPoints, 1), ...           % group
                    repmat(numericID, nPoints, 1), ...             % participant
                    repmat(trials, nPoints, 1), ...                % trial
                    repmat(forceLevels(force_levels), nPoints, 1), ... % forcelevel
                    timeVec(:), ...                                % time
                    apbVec(:), ...                                 % apb
                    fdsVec(:), ...                                 % fds
                    forceVec(:)                                    % force
                };
                
                % Append to main data collection
                dataCell = [dataCell; dataBlock];
            end
        end
    end
end

% Create table from collected data
dataTable = table(vertcat(dataCell{:,1}), vertcat(dataCell{:,2}), ...
                  vertcat(dataCell{:,3}), vertcat(dataCell{:,4}), ...
                  vertcat(dataCell{:,5}), vertcat(dataCell{:,6}), ...
                  vertcat(dataCell{:,7}), vertcat(dataCell{:,8}), ...
    'VariableNames', {'group', 'participant', 'trial', 'forcelevel', ...
                     'time', 'apb', 'fds', 'force'});

clearvars -except dataTable inputdata rms_data dataTable1
%% Write to .csv

% Write the table to a CSV file for R import
writetable(dataTable, 'rms_data_for_R_15.csv');

disp('Data converted to table and saved as rms_data_for_R.csv');