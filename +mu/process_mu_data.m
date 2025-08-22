function [analysed_mu_data, spiketrains] = process_mu_data()
    % PROCESS MU DATA
    % This function processes the decomposed motor unit spike train output 
    % from Delsys Neuromap software with force data to calculate 
    % motor unit metrics.
    %
    % Outputs:
    %   analysed_mu_data - A table containing the cleaned and labeled motor
    %                      unit metrics:
    %
    %                             * Motor unit average firing rate.
    %                             * Motor unit recruitment threshold.
    %                             * Accuracy (motor units with accuracy scores below 90%      
    %                               are removed).
    %                             * Coefficient of vatiation of interpulse
    %                               interval (IPI) (scores above 30% are removed).
    %                             * Number of IPIs used for the average firing
    %                               rate calculation (those with <3 are removed).
    %
    %   spiketrains     - A table containing firing times and metadata for all MUs:
    %                             * filename - Source file
    %                             * participant - Participant number
    %                             * force_level - Force level as % of MVC
    %                             * muscle - Muscle type
    %                             * trial - Trial number
    %                             * mu_idx - Motor unit index
    %                             * start_time - Analysis window start time
    %                             * firing_times - Cell array containing actual firing times
    %
    % Usage:
    %   [analysed_mu_data, spiketrains] = mu.data_analysis.process_mu_data();

    warning('off', 'MATLAB:table:ModifiedAndSavedVarnames'); % Turns off unnecessary warning

    % Configuration structure containing all path and grouping information
    CONFIG.FORCE_FOLDER = fullfile(pwd, '+coherence', '+emg_force_data');
    CONFIG.MU_FOLDER = fullfile(pwd, '+mu', '+decomposition_data');

    % Define participant groups for strength- and dexterity-trained
    CONFIG.GROUPS = struct(...
        'strength', [3, 4, 5, 6, 7, 8, 9, 11, 16, 20], ...   % Strength-trained participants
        'dexterity', [1, 2, 10, 12, 13, 14, 17, 18, 19, 21]); % Dexterity-trained participants

    fprintf('\nStep 1: Analysing force data...\n');
    [start_end_time, max_forces, force_data] = analyseForceData(CONFIG.FORCE_FOLDER);
    
    fprintf('\nStep 2: Creating timing data...\n');
    timing_data = createTimingData(start_end_time);
    
    fprintf('\nStep 3: Processing motor unit data...\n');
    [mu_data, spiketrains] = processMUData(CONFIG.MU_FOLDER, timing_data, force_data, max_forces);
    
    fprintf('\nStep 4: Cleaning and labeling data...\n');
    analysed_mu_data = cleanAndLabelData(mu_data, CONFIG.GROUPS);

    fprintf('\nProcessing complete. Data is output as analysed_mu_data and spiketrains.\n');
end


%% HELPER FUNCTIONS

function [start_end_time, max_forces, force_data] = analyseForceData(force_folder)
    % Wrapper for force analysis function
    % Returns timing windows, maximum forces, and processed force data
    [start_end_time, max_forces, force_data] = find_min_cov(force_folder);
end

function timing_data = createTimingData(results)
    % Creates standardised timing data structure from force analysis results
    timing_data = create_timing_data_from_results(results);
end

function [mu_data, spiketrains] = processMUData(mu_folder, timing_data, force_data, max_forces)
    % Processes all motor unit files with corresponding force and timing data
    [mu_data, spiketrains] = process_mu_files_with_thresholds(mu_folder, timing_data, force_data, max_forces);
end

function analysed_mu_data = cleanAndLabelData(data, groups)
    % Remove invalid data points based on quality criteria:
    % - Accuracy >= 90%
    % - Coefficient of variation of IPI <= 30%
    % - At least 7 interpulse intervals
    % - Positive firing threshold
    clean_data = data(data.accuracy >= 90 & ...
                      data.cov_ipi <= 30 & ...
                      data.num_ipi >= 7 & ...
                      data.firing_threshold > 0, :);
    
    % Sort data by force level and participant for easier analysis
    sorted_data = sortrows(clean_data, {'force_level', 'participant'});
    
    % Add group labels (strength vs dexterity) to each participant's data
    sorted_data.testing_group = repmat("", height(sorted_data), 1);  % Initialise empty group column
    sorted_data.testing_group(ismember(sorted_data.participant, groups.strength)) = "strength";
    sorted_data.testing_group(ismember(sorted_data.participant, groups.dexterity)) = "dexterity";

    % Reorder columns
    analysed_mu_data = sorted_data(:, {'participant', 'testing_group', 'force_level', 'trial', ...
    'muscle', 'mu_idx', 'accuracy', 'avg_firing_rate', 'cov_ipi', 'num_ipi', ...
    'firing_threshold', 'initial_firing_time', 'filename'});
end


function [start_end_time, max_forces, submax_force_data] = find_min_cov(force_folder_path)
    % Get list of all unique participant numbers from files
    fileList = dir(fullfile(force_folder_path, 'P*_MVC_Force.txt'));
    participantNums = zeros(length(fileList), 1);
    
    for i = 1:length(fileList)
        participantNums(i) = str2double(extractBetween(fileList(i).name, 'P', '_'));
    end
    
    % Sort participant numbers
    uniqueParticipants = sort(unique(participantNums));
    
    % Initialise storage structures
    start_end_time = struct();
    max_forces = table('Size', [length(uniqueParticipants), 2], ...
                      'VariableTypes', {'double', 'double'}, ...
                      'VariableNames', {'participant', 'maxforce'});
    submax_force_data = struct();
    
    % Force levels to process
    ForceLevels = [15, 35, 55, 70];
    
    % Analysis parameters
    window_duration = 4000;
    step_size = 50;
    sampling_rate = 2000;
    start_time_range = 7;
    end_time_range = 13;
    
    % Process each participant
    for p = 1:length(uniqueParticipants)
        participantNum = uniqueParticipants(p);
        fprintf('Processing participant P%d...\n', participantNum);
        
        % Create participant field
        participant_field = sprintf('P%d', participantNum);
        
        % Load and process MVC force
        mvcFilename = fullfile(force_folder_path, sprintf('P%d_MVC_Force.txt', participantNum));
        if ~isfile(mvcFilename)
            warning('MVC file not found for P%d, skipping...', participantNum);
            continue;
        end
        
        MVCforce = readtable(mvcFilename);
        MVCforce = table2array(MVCforce);
        rowsWithNaN = any(isnan(MVCforce), 2);
        MVCforce = MVCforce(~rowsWithNaN, :);
        maxforce = max(MVCforce(:,2));
        
        % Store max force with participant number
        max_forces.participant(p) = participantNum;
        max_forces.maxforce(p) = maxforce;
        
        % Calculate target force levels
        target = maxforce * [0.15, 0.35, 0.55, 0.70]';
        
        % Initialise participant structure in submax_force_data
        submax_force_data.(participant_field) = struct();
        
        % Load submaximal files
        for i = 1:length(ForceLevels)
            % Create force level field
            force_field = sprintf('MVC%d', ForceLevels(i));
            submax_force_data.(participant_field).(force_field) = struct();
            
            subFilename = fullfile(force_folder_path, sprintf('P%d_%dMVC_Force.txt', participantNum, ForceLevels(i)));
            
            if ~isfile(subFilename)
                warning('Submaximal file not found: %s', subFilename);
                continue;
            end
            
            % Read and process force data
            force = readtable(subFilename);
            force = table2array(force);
            
            % Split force data into trials and store in structure
            submax_force_data.(participant_field).(force_field).rep1 = force(3:40002, :);
            submax_force_data.(participant_field).(force_field).rep2 = force(40012:80011, :);
            submax_force_data.(participant_field).(force_field).rep3 = force(80021:120020, :);
            
            % Initialise trials structure for results
            trials_struct = struct();
            
            % Process each trial for results
            for j = 1:3
                % Get the appropriate force data based on repetition
                switch j
                    case 1
                        trial_force = force(3:40002, 2);
                    case 2
                        trial_force = force(40012:80011, 2);
                    case 3
                        trial_force = force(80021:120020, 2);
                end
                
                % Calculate sliding window COV
                results = sliding_window_cov(trial_force, target(i), window_duration, step_size, sampling_rate);
                
                % Find minimum COV within time range
                filtered_results = results(results(:,1) >= start_time_range & results(:,2) <= end_time_range, :);
                
                if ~isempty(filtered_results)
                    [min_COV, min_index] = min(filtered_results(:,3));
                    min_start_time = filtered_results(min_index, 1);
                    min_end_time = filtered_results(min_index, 2);
                    
                    % Store results for this trial
                    trials_struct.(sprintf('trial%d', j)) = struct(...
                        'start_time', min_start_time, ...
                        'end_time', min_end_time, ...
                        'min_COV', min_COV);
                else
                    trials_struct.(sprintf('trial%d', j)) = struct(...
                        'start_time', NaN, ...
                        'end_time', NaN, ...
                        'min_COV', NaN);
                end
            end
            
            % Store trials structure in force level field
            start_end_time.(participant_field).(force_field) = trials_struct;
        end
    end
end

function results = sliding_window_cov(force_data, target_force, window_duration, step_size, sampling_rate)
% Function for sliding window analysis

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
        cov_value = std_force / target_force;
        
        start_time = (start_idx - 1) / sampling_rate;
        end_time = (end_idx - 1) / sampling_rate;
        
        results(i, :) = [start_time, end_time, cov_value];
    end
end



%% DATA ANALYSIS FUNCTIONS

function [MU_data, unit_firing_times] = calculate_firing_rate(input_decomp_data, start_time, end_time, force_data, max_force)
    % Calculates firing rate metrics for each motor unit
    % 
    % Inputs:
    %   input_decomp_data - Structure from Delsys Neuromap containing decomposed MU firing times
    %   start_time - Start time of analysis window
    %   end_time - End time of analysis window
    %   force_data - Matrix of force measurements [time, force]
    %   max_force - Maximum force value for normalisation
    %
    % Outputs:
    %   MU_data - Table with calculated MU metrics
    %   unit_firing_times - Cell array containing the firing times for each MU
    
    % Get total number of motor units
    num_MUs = length(input_decomp_data.MUFir{1,1});
    
    % Initialise structure to hold all metrics and cell array for firing times
    metrics = initialiseMetrics(num_MUs);
    unit_firing_times = cell(num_MUs, 1);
    
    % Process each motor unit individually
    for mu_idx = 1:num_MUs
        % Extract firing times for current motor unit
        firing_times = input_decomp_data.MUFir{1,1}{mu_idx,1};
        
        % Store the complete firing times for this unit
        unit_firing_times{mu_idx} = firing_times;
        
        % Calculate all metrics for this motor unit
        metrics = processSingleUnit(metrics, mu_idx, firing_times, start_time, end_time, ...
                                 force_data, max_force, input_decomp_data);
    end
    
    % Convert results to table format
    MU_data = createOutputTable(metrics);
end

function metrics = initialiseMetrics(num_units)
    % Initialise empty structure for storing all motor unit metrics
    metrics = struct();
    
    % Define all fields to be tracked
    fields = {'avg_firing_rate', 'cov_ipi', 'num_ipi', 'accuracy', ...
              'firing_threshold', 'initial_firing_time', 'mu_idx'};  % Added mu_idx here
    
    % Create zero arrays for each metric
    for field = fields
        metrics.(field{1}) = zeros(num_units, 1);
    end
end

function metrics = processSingleUnit(metrics, idx, firing_times, start_time, end_time, ...
                                   force_data, max_force, decomp_data)
    % Store the motor unit index
    metrics.mu_idx(idx) = idx;  % Add this line to store the index

    % Store time of first firing for recruitment order analysis
    metrics.initial_firing_time(idx) = firing_times(1);
    
    % Calculate force level at which unit begins firing consistently
    metrics.firing_threshold(idx) = calculateThreshold(firing_times, force_data, max_force);
    
    % Calculate firing pattern metrics (rate, variability, number of intervals)
    [rate, cov, num] = calculateFiringMetrics(firing_times, start_time, end_time);
    metrics.avg_firing_rate(idx) = rate;
    metrics.cov_ipi(idx) = cov;
    metrics.num_ipi(idx) = num;
    
    % Store decomposition accuracy for quality control
    metrics.accuracy(idx) = decomp_data.Stats{1,1}{idx,1};
end

function threshold = calculateThreshold(firing_times, force_data, max_force)
    % Find time point where firing becomes consistent (COV below threshold)
    time_below_cov = get_sliding_cov(firing_times);
    
    if isempty(time_below_cov)
        threshold = NaN;
        return;
    end
    
    % Find force level at threshold time point
    [~, idx] = min(abs(force_data(:,1) - time_below_cov));
    threshold = (force_data(idx, 2) / max_force) * 100;  % Convert to % MVC
end

function [rate, cov, num] = calculateFiringMetrics(firing_times, start_time, end_time)
    % Extract firing times within analysis window
    filtered_times = firing_times(firing_times >= start_time & firing_times <= end_time);
    
    % Calculate inter-pulse intervals (time between consecutive firings)
    IPIs = diff(filtered_times);
    
    if isempty(IPIs)
        % Handle case with insufficient data points
        rate = NaN; 
        cov = NaN; 
        num = 0;
        return;
    end
    
    % Calculate metrics:
    avg_IPI = mean(IPIs);
    rate = mean(1./IPIs);                          % Convert IPI to frequency
    cov = (std(IPIs') / avg_IPI) * 100;         % Coefficient of variation
    num = length(IPIs');                         % Number of intervals
end

function table_out = createOutputTable(metrics)
    % Convert metrics structure to organised table format
    table_out = table(metrics.mu_idx, metrics.accuracy, metrics.avg_firing_rate, ...  % Added mu_idx here
                     metrics.cov_ipi, metrics.num_ipi, ...
                     metrics.firing_threshold, metrics.initial_firing_time, ...
                     'VariableNames', {'mu_idx', 'accuracy', 'avg_firing_rate', 'cov_ipi', ...  % Added mu_idx here
                                     'num_ipi', 'firing_threshold', 'initial_firing_time'});
end

function [all_MU_data, spiketrains] = process_mu_files_with_thresholds(mu_folder, timing_data, force_data, max_forces)
    % Process all motor unit files and calculate firing thresholds
    % 
    % Inputs:
    %   mu_folder - Path to folder containing motor unit data files
    %   timing_data - Table with start/end times for each trial
    %   force_data - Structure containing force measurements
    %   max_forces - Table of maximum forces for each participant
    %
    % Outputs:
    %   all_MU_data - Table with all processed motor unit metrics
    %   spiketrains - Table with firing time data and metadata
    
    % Get list of all .mat files in folder
    files = dir(fullfile(mu_folder, '*.mat'));
    
    % Initialise empty table for results and cell arrays for spiketrains data
    all_MU_data = table();
    
    % Preallocate cell arrays for spiketrains table columns
    total_files = length(files) * 10;  % Rough estimate of total MUs (assuming ~10 per file)
    filenames = cell(total_files, 1);
    participants = zeros(total_files, 1);
    force_levels = zeros(total_files, 1);
    muscles = cell(total_files, 1);
    trials = zeros(total_files, 1);
    mu_indices = zeros(total_files, 1);
    start_times = zeros(total_files, 1);
    end_times = zeros(total_files, 1);  % Add end_time array
    firing_times_array = cell(total_files, 1);
    
    % Track the current index in the spiketrains arrays
    current_idx = 1;
    
    % Process each file
    for i = 1:length(files)
        try
            % Process single file and append results
            [result, current_file_data, num_units] = processSingleFile(files(i), timing_data, force_data, max_forces, mu_folder);
            all_MU_data = [all_MU_data; result];
            
            % Extract data for spiketrains table
            for j = 1:num_units
                % If we need more space, resize the arrays
                if current_idx > length(filenames)
                    % Double the size of all arrays
                    filenames = [filenames; cell(length(filenames), 1)];
                    participants = [participants; zeros(length(participants), 1)];
                    force_levels = [force_levels; zeros(length(force_levels), 1)];
                    muscles = [muscles; cell(length(muscles), 1)];
                    trials = [trials; zeros(length(trials), 1)];
                    mu_indices = [mu_indices; zeros(length(mu_indices), 1)];
                    start_times = [start_times; zeros(length(start_times), 1)];
                    end_times = [end_times; zeros(length(end_times), 1)];  % Resize end_times too
                    firing_times_array = [firing_times_array; cell(length(firing_times_array), 1)];
                end
                
                % Store data for this motor unit
                filenames{current_idx} = current_file_data.filename;
                participants(current_idx) = current_file_data.participant;
                force_levels(current_idx) = current_file_data.force_level;
                muscles{current_idx} = current_file_data.muscle;
                trials(current_idx) = current_file_data.trial;
                mu_indices(current_idx) = j;
                start_times(current_idx) = current_file_data.start_time;
                end_times(current_idx) = current_file_data.end_time;  % Store end_time
                firing_times_array{current_idx} = current_file_data.firing_times{j};
                
                current_idx = current_idx + 1;
            end
            
        catch ME
            warning('Error processing %s: %s', files(i).name, ME.message);
        end
    end
    
    % Trim arrays to actual size used
    actual_size = current_idx - 1;
    filenames = filenames(1:actual_size);
    participants = participants(1:actual_size);
    force_levels = force_levels(1:actual_size);
    muscles = muscles(1:actual_size);
    trials = trials(1:actual_size);
    mu_indices = mu_indices(1:actual_size);
    start_times = start_times(1:actual_size);
    end_times = end_times(1:actual_size);  % Trim end_times
    firing_times_array = firing_times_array(1:actual_size);
    
    % Create table from arrays
    spiketrains = table(filenames, participants, force_levels, muscles, ...
                        trials, mu_indices, start_times, end_times, firing_times_array, ...
                        'VariableNames', {'filename', 'participant', 'force_level', ...
                                         'muscle', 'trial', 'mu_idx', 'start_time', 'end_time', 'firing_times'});
end

function [result, file_data, num_units] = processSingleFile(file, timing_data, force_data, max_forces, mu_folder)
    % Extract metadata from filename
    [participant_num, force_level, trial_num, muscle] = parseFilename(file.name);
    
    % Verify all required information was found
    validateInputs(participant_num, force_level, trial_num, muscle);
    
    % Get timing information for this trial
    [start_time, end_time] = getTiming(timing_data, file.name);
    
    % Get corresponding force data
    [force_profile, max_force] = getForceData(force_data, max_forces, ...
                                            participant_num, force_level, trial_num);
    
    % Load and process motor unit data
    decomp_data = load(fullfile(mu_folder, file.name));
    [MU_data, unit_firing_times] = calculate_thresholds(decomp_data, force_profile, max_force, start_time, end_time);
    
    % Get the number of motor units in this file
    num_units = length(unit_firing_times);
    
    % Add metadata to results
    result = addMetadata(MU_data, file.name, participant_num, force_level, trial_num, muscle);
    
    % Create data structure for this file
    file_data = struct();
    file_data.filename = file.name;
    file_data.participant = participant_num;
    file_data.force_level = force_level;
    file_data.muscle = muscle;
    file_data.trial = trial_num;
    file_data.start_time = start_time;
    file_data.end_time = end_time;  % Add end_time
    file_data.firing_times = unit_firing_times;
end

function [participant_num, force_level, trial_num, muscle] = parseFilename(filename)
    % Extract participant number (e.g., from P1_15mvc_rep1_apb.mat)
    p_match = regexp(filename, 'P(\d+)', 'tokens');
    
    % Extract force level (e.g., 15 from 15mvc)
    mvc_match = regexp(filename, '(\d+)mvc', 'tokens');
    
    % Extract repetition/trial number
    rep_match = regexp(filename, 'rep(\d+)', 'tokens');
    
    % Convert extracted strings to numbers
    participant_num = str2double(p_match{1}{1});
    force_level = str2double(mvc_match{1}{1});
    trial_num = str2double(rep_match{1}{1});
    
    % Determine muscle type from filename
    if contains(filename, '_apb')
        muscle = 'apb';  % Abductor Pollicis Brevis
    elseif contains(filename, '_fds')
        muscle = 'fds';  % Flexor Digitorum Superficialis
    else
        muscle = '';
    end
end

function validateInputs(participant_num, force_level, trial_num, muscle)
    % Verify all required information was successfully extracted
    if isempty(participant_num) || isempty(force_level) || ...
       isempty(trial_num) || isempty(muscle)
        error('Invalid filename format');
    end
end

function [start_time, end_time] = getTiming(timing_data, filename)
    % Find matching timing data for current file
    idx = strcmpi(timing_data.filename, filename);
    
    if ~any(idx)
        error('No timing data found');
    end
    
    % Extract start and end times
    start_time = timing_data.start_time(idx);
    end_time = timing_data.end_time(idx);
end

function [force_profile, max_force] = getForceData(force_data, max_forces, ...
                                                 participant_num, force_level, trial_num)
    % Create field names for accessing nested structure
    participant_field = sprintf('P%d', participant_num);
    force_field = sprintf('MVC%d', force_level);
    rep_field = sprintf('rep%d', trial_num);
    
    % Get force profile for this trial
    force_profile = force_data.(participant_field).(force_field).(rep_field);
    
    % Get maximum force for participant
    max_force_idx = max_forces.participant == participant_num;
    
    if ~any(max_force_idx)
        error('No max force found for participant %d', participant_num);
    end
    max_force = max_forces.maxforce(max_force_idx);
end

function result = addMetadata(MU_data, filename, participant_num, force_level, trial_num, muscle)
    % Add file information to each row of results
    % Creates uniform metadata columns for all motor units in the file
    MU_data.filename = repmat(string(filename), height(MU_data), 1);
    MU_data.participant = repmat(participant_num, height(MU_data), 1);
    MU_data.force_level = repmat(force_level, height(MU_data), 1);
    MU_data.trial = repmat(trial_num, height(MU_data), 1);
    MU_data.muscle = repmat(string(muscle), height(MU_data), 1);
    
    result = MU_data;
end

function timing_data = create_timing_data_from_results(results_struct)
    % Initialise arrays to store the data
    filenames = {};
    start_times = [];
    end_times = [];
    
    % Force levels to process
    force_levels = {'15', '35', '55', '70'};
    
    % Muscles to process
    muscles = {'apb', 'fds'};
    
    % Get all participant fields
    participant_fields = fieldnames(results_struct);
    
    % Loop through each participant
    for p = 1:length(participant_fields)
        participant = participant_fields{p};
        
        % Loop through each force level
        for f = 1:length(force_levels)
            mvc_field = ['MVC' force_levels{f}];
            
            % Loop through each trial
            for t = 1:3
                trial_field = ['trial' num2str(t)];
                
                % Get the timing data
                timing = results_struct.(participant).(mvc_field).(trial_field);
                
                % Create entries for both muscles
                for m = 1:length(muscles)
                    % Create filename in required format (e.g., 'P21_70MVC_rep2_fds.mat')
                    filename = sprintf('%s_%smvc_rep%d_%s.mat', ...
                        participant, ...
                        force_levels{f}, ...
                        t, ...
                        muscles{m});
                    
                    % Add to arrays
                    filenames{end+1} = filename;
                    start_times(end+1) = timing.start_time;
                    end_times(end+1) = timing.end_time;
                end
            end
        end
    end
    
    % Create table
    timing_data = table(string(filenames)', start_times', end_times', ...
        'VariableNames', {'filename', 'start_time', 'end_time'});
end

function [MU_data, unit_firing_times] = calculate_thresholds(input_decomp_data, force_data, maxforce, start_time, end_time)
    % Calculate basic metrics first using the existing function with modified output
    [MU_data, unit_firing_times] = calculate_firing_rate(input_decomp_data, start_time, end_time, force_data, maxforce);
    
    % Add new column for threshold percentage
    num_motor_units = length(input_decomp_data.MUFir{1,1});
    firing_threshold_perc = zeros(num_motor_units, 1);
    
    % Process each motor unit
    for j = 1:num_motor_units
        % Get firing threshold using sliding window
        time_below_cov = get_sliding_cov(input_decomp_data.MUFir{1,1}{j,1});
        
        % Check if valid time was found
        if isempty(time_below_cov)
            firing_threshold_perc(j) = NaN;
        else
            % Find the closest time in force data and get corresponding force value
            [~, idx_closest] = min(abs(force_data(:,1) - time_below_cov));
            threshold_force = force_data(idx_closest, 2);
            firing_threshold_perc(j) = (threshold_force / maxforce) * 100;
        end
    end
    
end


function time = get_sliding_cov(times)
% get_sliding_cov computes the coefficient of variation (COV) for sliding windows
% over the inter-spike intervals (ISIs) calculated from the input array of
% firing times. The function slides a 500 ms (0.5 sec) window over the ISIs,
% calculates the COV for each window, and returns the time corresponding to
% the first window where COV is less than 0.3.
%
% INPUT
% 
%      times   : 1 x n array of firing times (in seconds)
%      
% OUTPUT
% 
%      time    : Time corresponding to the first window where COV < 0.3
%                (empty if no such time is found)
%

    % Initialise parameters
    window_size = 0.75;  % 750 ms = 0.75 sec window
    COV_threshold = 0.3;  % COV threshold for detection

    % Ensure the input is a vector
    if ~isvector(times)
        error('times must be a vector');
    end
    
    % Calculate inter-spike intervals (ISIs)
    ISIs = diff(times);  % Calculate differences between consecutive spikes
    
    % Initialise time as empty in case no valid COV < 0.3 is found
    time = [];

    % Iterate over the time array with a sliding window
    for i = 1:length(times)-1
        % Find the window end time based on spike times
        window_end_time = times(i) + window_size;

        % Extract the current window of ISIs corresponding to spike times in this window
        window_ISIs = ISIs(times(1:end-1) >= times(i) & times(1:end-1) < window_end_time);
        
        % Ensure we have enough ISIs to calculate COV
        if length(window_ISIs) >= 3  % COV needs at least three ISIs
            % Calculate the coefficient of variation for this window
            COV = getcov(window_ISIs);

            % Check if the COV is below the threshold
            if COV < COV_threshold
                time = times(i);  % Assign the time corresponding to the start of the window
                return;            % Return immediately once the condition is met
            end
        end
    end
end

function COV = getcov(x)
% Function to calculate coefficient of variation
    COV = std(x)/mean(x);
end
