function [combined_table, comparison_table, histogram_table] = export_coherenceData_for_R(pooled_coherence, comparison_of_coherence, pooled_participant_coherence)
% CREATE_COHERENCE_TABLES Creates tables of coherence and comparison data
%   [combined_table, comparison_table, histogram_table] = plotting_data_for_R(pooled_coherence, comparison_of_coherence, pooled_participant_coherence)
%
%   Takes the pooled_coherence and comparison_of_coherence structures from calculate_coherence()
%   and creates .csv files for plotting in R. Also creates a histogram table with group and
%   individual participant data.
%   
%   
% Input:
%   pooled_coherence - Structure containing pooled coherence results
%   comparison_of_coherence - Structure containing comparison results between groups
%   pooled_participant_coherence - Structure containing individual participant coherence results
%
% Output:
%   combined_table - Table containing coherence results for both groups
%   comparison_table - Table containing comparison of coherence results
%   histogram_table - Table containing histogram data for groups and individual participants

% Get frequencies from MVC15 (same for all tables)
freq = pooled_coherence.strength.MVC15.f(:,1);

% Initialize variables
mvc_levels = {'15', '35', '55', '70'};
n_freqs = length(freq);

% Get participant names from the structure fieldnames
strength_participants = fieldnames(pooled_participant_coherence.strength);
dexterity_participants = fieldnames(pooled_participant_coherence.dexterity);

%% Create combined coherence table
% Create variable names for the combined table
var_names = {'freq'};
% Add strength columns first
for mvc = mvc_levels
    var_names = [var_names, ...
        {['strength_coh_' mvc{1}], ...
         ['strength_ci_' mvc{1}], ...
         ['strength_chi2_sig_' mvc{1}]}];
end
% Then add dexterity columns
for mvc = mvc_levels
    var_names = [var_names, ...
        {['dexterity_coh_' mvc{1}], ...
         ['dexterity_ci_' mvc{1}], ...
         ['dexterity_chi2_sig_' mvc{1}]}];
end

% Initialize data matrix for combined table
% Column 1: freq
% Columns 2-13: all strength data (3 columns per MVC level)
% Columns 14-25: all dexterity data (3 columns per MVC level)
combined_data = zeros(n_freqs, length(mvc_levels) * 6 + 1);
combined_data(:,1) = freq;

% Fill strength data
for i = 1:length(mvc_levels)
    mvc_key = ['MVC' mvc_levels{i}];
    col_idx = (i-1)*3 + 2; % Starting column for this MVC level
    
    % Strength group data
    combined_data(:,col_idx) = pooled_coherence.strength.(mvc_key).f(:,4); % Coherence
    combined_data(:,col_idx+1) = repmat(pooled_coherence.strength.(mvc_key).cl.ch_c95, n_freqs, 1); % CI
    
    % Binary chi2 test result: 1 if significant, 0 if not
    chi2_values = pooled_coherence.strength.(mvc_key).f(:,8);
    chi2_threshold = repmat(pooled_coherence.strength.(mvc_key).cl.chi_c95, n_freqs, 1);
    combined_data(:,col_idx+2) = double(chi2_values >= chi2_threshold); % Binary result (0 or 1)
end

% Fill dexterity data
for i = 1:length(mvc_levels)
    mvc_key = ['MVC' mvc_levels{i}];
    col_idx = length(mvc_levels)*3 + (i-1)*3 + 2; % Starting column for this MVC level
    
    % Dexterity group data
    combined_data(:,col_idx) = pooled_coherence.dexterity.(mvc_key).f(:,4); % Coherence
    combined_data(:,col_idx+1) = repmat(pooled_coherence.dexterity.(mvc_key).cl.ch_c95, n_freqs, 1); % CI
    
    % Binary chi2 test result: 1 if significant, 0 if not
    chi2_values = pooled_coherence.dexterity.(mvc_key).f(:,8);
    chi2_threshold = repmat(pooled_coherence.dexterity.(mvc_key).cl.chi_c95, n_freqs, 1);
    combined_data(:,col_idx+2) = double(chi2_values >= chi2_threshold); % Binary result (0 or 1)
end

% Create combined table
combined_table = array2table(combined_data, 'VariableNames', var_names);

%% Create comparison table
% Create variable names for the comparison table
comparison_var_names = {'freq'};
for mvc = mvc_levels
    comparison_var_names = [comparison_var_names, ...
        {['compcoh_' mvc{1}], ...  % Comparison of coherence column
         ['ci_' mvc{1}]}];         % Confidence interval column
end

% Initialize data matrix for comparison table
comparison_data = zeros(n_freqs, length(mvc_levels) * 2 + 1);
comparison_data(:,1) = freq;

% Fill comparison data
for i = 1:length(mvc_levels)
    mvc_key = ['MVC' mvc_levels{i}];
    col_idx = i*2; % Starting column for this MVC level
    
    % Comparison data
    comparison_data(:,col_idx) = comparison_of_coherence.(mvc_key).f(:,2);     % Comparison of coherence
    comparison_data(:,col_idx+1) = repmat(comparison_of_coherence.(mvc_key).cl.cmpcoh_c95, n_freqs, 1); % CI
end

% Create comparison table
comparison_table = array2table(comparison_data, 'VariableNames', comparison_var_names);

%% Create histogram table
% Create variable names for the histogram table
histogram_var_names = {'freq'};

% Add group column names
groups = {'strength', 'dexterity'};
for g = 1:length(groups)
    group = groups{g};
    for mvc = mvc_levels
        histogram_var_names{end+1} = [group '_MVC' mvc{1}];
    end
end

% Add participant column names
% First strength participants
for p = 1:length(strength_participants)
    participant = strength_participants{p}; % Already in format 'P1', 'P3', etc.
    for i = 1:length(mvc_levels)
        mvc_key = mvc_levels{i};
        histogram_var_names{end+1} = [participant '_MVC' mvc_key];
    end
end

% Then dexterity participants
for p = 1:length(dexterity_participants)
    participant = dexterity_participants{p}; % Already in format 'P2', 'P10', etc.
    for i = 1:length(mvc_levels)
        mvc_key = mvc_levels{i};
        histogram_var_names{end+1} = [participant '_MVC' mvc_key];
    end
end

% Calculate total number of columns in histogram table
% 1 for frequency + 8 for groups (2 groups x 4 MVC levels) + 80 for participants (20 participants x 4 MVC levels)
n_histogram_cols = 1 + (length(groups) * length(mvc_levels)) + ((length(strength_participants) + length(dexterity_participants)) * length(mvc_levels));

% Initialize data matrix for histogram table
histogram_data = zeros(n_freqs, n_histogram_cols);
histogram_data(:,1) = freq;

% Fill group histogram data
col_idx = 2;
for g = 1:length(groups)
    group = groups{g};
    for i = 1:length(mvc_levels)
        mvc_key = ['MVC' mvc_levels{i}];
        
        % Extract histogram data from pooled_coherence
        histogram_data(:,col_idx) = pooled_coherence.(group).(mvc_key).f(:,7);
        
        col_idx = col_idx + 1;
    end
end

% Fill participant histogram data
% First strength participants
for p = 1:length(strength_participants)
    participant = strength_participants{p}; % Already in format 'P1', 'P3', etc.
    for i = 1:length(mvc_levels)
        mvc_key = ['MVC' mvc_levels{i}];
        
        % Extract histogram data from pooled_participant_coherence
        histogram_data(:,col_idx) = pooled_participant_coherence.strength.(participant).(mvc_key).f(:,7);
        
        col_idx = col_idx + 1;
    end
end

% Then dexterity participants
for p = 1:length(dexterity_participants)
    participant = dexterity_participants{p}; % Already in format 'P2', 'P10', etc.
    for i = 1:length(mvc_levels)
        mvc_key = ['MVC' mvc_levels{i}];
        
        % Extract histogram data from pooled_participant_coherence
        histogram_data(:,col_idx) = pooled_participant_coherence.dexterity.(participant).(mvc_key).f(:,7);
        
        col_idx = col_idx + 1;
    end
end

% Create histogram table
histogram_table = array2table(histogram_data, 'VariableNames', histogram_var_names);

end