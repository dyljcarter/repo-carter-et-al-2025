function neurospec_plot_coherence(pooled_coherence, comparison)
% NEUROSPEC_PLOT_COHERENCE Generate visualization plots for coherence analysis results
%
% This function creates three sets of plots:
%   1. Coherence comparison between groups
%   2. Strength-trained pooled coherence 
%   3. Dexterity-trained pooled coherence
%
% Parameters:
%   pooled_coherence - Structure containing pooled coherence results with fields:
%                      .strength and .dexterity, each containing MVC15, MVC35, etc.
%   comparison - Structure containing statistical comparison between groups
%
% Returns: 
%   Comparison of coherence            - Figures 1-4
%   Strength-trained pooled coherence  - Figures 5-8
%   Dexterity-trained pooled coherence - Figures 9-12
%
% Example usage:
%   neurospec_plot_coherence(pooled_coherence, comparison);

% Input validation
if ~isstruct(pooled_coherence) || ~isfield(pooled_coherence, 'strength') || ~isfield(pooled_coherence, 'dexterity')
    error('pooled_coherence must be a structure with fields ''strength'' and ''dexterity''');
end
if ~isstruct(comparison)
    error('comparison must be a structure');
end

% Plot configuration
PLOT_CONFIG = struct();
PLOT_CONFIG.FORCE_LEVELS = {'15', '35', '55', '70'};  % Force levels as % of MVC
PLOT_CONFIG.MAX_FREQUENCY = 70;       % Maximum frequency for plots (Hz)
PLOT_CONFIG.MAX_COHERENCE = 0.22;     % Maximum coherence value for comparison plots
PLOT_CONFIG.FIGURE_SIZE = [600, 400]; % Figure size [width, height]
PLOT_CONFIG.TIME_RANGE = 1;           % Time range for cumulant plots (ms)
PLOT_CONFIG.TIME_NEGATIVE = 0.5;      % Negative time range for cumulant plots (ms)
PLOT_CONFIG.COHERENCE_MAX = 0.05;     % Maximum coherence for pooled plots
PLOT_CONFIG.CHI_MAX = 30;             % Maximum chi-squared value

% Create force level labels
force_labels = cellfun(@(x) sprintf('MVC%s', x), PLOT_CONFIG.FORCE_LEVELS, 'UniformOutput', false);

% Plot coherence comparisons
for i = 1:length(force_labels)
    figure(i);
    set(gcf, 'Position', [100, 100, PLOT_CONFIG.FIGURE_SIZE]);
    psp_compcoh1(comparison.(force_labels{i}).f, ...
        comparison.(force_labels{i}).cl, ...
        PLOT_CONFIG.MAX_FREQUENCY, ...
        PLOT_CONFIG.MAX_COHERENCE, ...
        sprintf('Comparison of Coherence at %s%% MVC', PLOT_CONFIG.FORCE_LEVELS{i}));
end

% Plot strength group pooled coherence
for i = 1:length(force_labels)
    figure(i + length(force_labels));
    set(gcf, 'Position', [100, 100, PLOT_CONFIG.FIGURE_SIZE]);
    psp2_pool6(pooled_coherence.strength.(force_labels{i}).f, ...
        pooled_coherence.strength.(force_labels{i}).t, ...
        pooled_coherence.strength.(force_labels{i}).cl, ...
        PLOT_CONFIG.MAX_FREQUENCY, ...
        PLOT_CONFIG.TIME_RANGE, ...
        PLOT_CONFIG.TIME_NEGATIVE, ...
        PLOT_CONFIG.COHERENCE_MAX, ...
        PLOT_CONFIG.CHI_MAX);
    sgtitle(sprintf('Strength-Trained Pooled Coherence at %s%% MVC', PLOT_CONFIG.FORCE_LEVELS{i}));
end

% Plot dexterity group pooled coherence
for i = 1:length(force_labels)
    figure(i + 2 * length(force_labels));
    set(gcf, 'Position', [100, 100, PLOT_CONFIG.FIGURE_SIZE]);
    psp2_pool6(pooled_coherence.dexterity.(force_labels{i}).f, ...
        pooled_coherence.dexterity.(force_labels{i}).t, ...
        pooled_coherence.dexterity.(force_labels{i}).cl, ...
        PLOT_CONFIG.MAX_FREQUENCY, ...
        PLOT_CONFIG.TIME_RANGE, ...
        PLOT_CONFIG.TIME_NEGATIVE, ...
        PLOT_CONFIG.COHERENCE_MAX, ...
        PLOT_CONFIG.CHI_MAX);
    sgtitle(sprintf('Dexterity-Trained Pooled Coherence at %s%% MVC', PLOT_CONFIG.FORCE_LEVELS{i}));
end
end