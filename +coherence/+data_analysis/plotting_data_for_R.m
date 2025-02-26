function [combined_table, comparison_table] = plotting_data_for_R(pooled_coherence, comparison_of_coherence)
% CREATE_COHERENCE_TABLES Creates tables of coherence and comparison data
%   [combined_table, comparison_table] = plotting_data_for_R(pooled_coherence, comparison_of_coherence)
%
%   Takes the pooled_coherence and comparison_of_coherence structures from calculate_coherence()
%   and creates .csv files for plotting in R.
%   
%   
% Input:
%   pooled_coherence - Structure containing pooled coherence results
%   comparison_of_coherence - Structure containing comparison results between groups
%
% Output:
%   combined_table - Table containing coherence results for both groups
%   comparison_table - Table containing comparison of coherence results

% Get frequencies from MVC15 (same for both tables)
freq = pooled_coherence.strength.MVC15.f(:,1);

% Initialize variables
mvc_levels = {'15', '35', '55', '70'};
n_freqs = length(freq);

%% Create combined coherence table
% Create variable names for the combined table
var_names = {'freq'};
% Add strength columns first
for mvc = mvc_levels
    var_names = [var_names, ...
        {['strength_coh_' mvc{1}], ...
         ['strength_ci_' mvc{1}]}];
end
% Then add dexterity columns
for mvc = mvc_levels
    var_names = [var_names, ...
        {['dexterity_coh_' mvc{1}], ...
         ['dexterity_ci_' mvc{1}]}];
end

% Initialize data matrix for combined table
% Column 1: freq
% Columns 2-9: all strength data
% Columns 10-17: all dexterity data
combined_data = zeros(n_freqs, length(mvc_levels) * 4 + 1);
combined_data(:,1) = freq;

% Fill strength data
for i = 1:length(mvc_levels)
    mvc_key = ['MVC' mvc_levels{i}];
    col_idx = i*2; % Starting column for this MVC level
    
    % Strength group data
    combined_data(:,col_idx) = pooled_coherence.strength.(mvc_key).f(:,4); % Coherence
    combined_data(:,col_idx+1) = repmat(pooled_coherence.strength.(mvc_key).cl.ch_c95, n_freqs, 1); % CI
end

% Fill dexterity data
for i = 1:length(mvc_levels)
    mvc_key = ['MVC' mvc_levels{i}];
    col_idx = length(mvc_levels)*2 + i*2; % Starting column for this MVC level
    
    % Dexterity group data
    combined_data(:,col_idx) = pooled_coherence.dexterity.(mvc_key).f(:,4); % Coherence
    combined_data(:,col_idx+1) = repmat(pooled_coherence.dexterity.(mvc_key).cl.ch_c95, n_freqs, 1); % CI
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

end