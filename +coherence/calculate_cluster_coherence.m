%% Calculate pooled coherence analysis for participants in cluster 1

% Load data
load("+coherence/input_for_cluster_coherence.mat")

%% Run for 15% MVC
% Input force level
config.FORCE_LEVELS = {'15'};  % Force levels as % of MVC

% Number of repetitions/trials per force level
config.REPS = {'1', '2', '3'};  % Number of repetitions

% Specify which participants to include in each group
config.PARTICIPANTS = struct();
config.PARTICIPANTS.dexterity = {'P19', 'P21'};
config.PARTICIPANTS.strength = {'P3', 'P4', 'P6', 'P8', 'P9', 'P16'};

% Run function to calculate coherence for cluster 1
clust_1_15mvc = calculate_cluster_pooled_coherence(coherence_data, config);


%% Run for 35% MVC
% Input force level
config.FORCE_LEVELS = {'15','35'};  % Force levels as % of MVC

% Number of repetitions/trials per force level
config.REPS = {'1', '2', '3'};  % Number of repetitions

% Specify which participants to include in each group
config.PARTICIPANTS = struct();
config.PARTICIPANTS.dexterity = {'P1', 'P2', 'P13', 'P18', 'P19', 'P21'};
config.PARTICIPANTS.strength = {'P3', 'P4', 'P6', 'P8', 'P9', 'P16', 'P20'};

clust_1_35mvc = calculate_cluster_pooled_coherence(coherence_data, config);

%% CREATE TABLE

sources = struct(...
    'data', {clust_1_15mvc.dexterity.MVC15, clust_1_15mvc.strength.MVC15, ...
             clust_1_35mvc.dexterity.MVC35, clust_1_35mvc.strength.MVC35}, ...
    'name', {'clust1_15mvc_dext', 'clust1_15mvc_str', ...
             'clust1_35mvc_dext', 'clust1_35mvc_str'});

% Initialize table
cluster_coherence = table();
cluster_coherence.f = clust_1_35mvc.dexterity.MVC35.f(:,1);
n_rows = height(cluster_coherence);

% Add all columns
for i = 1:length(sources)
    cluster_coherence.(sources(i).name) = sources(i).data.f(:,4);
    cluster_coherence.(['ci_' sources(i).name]) = repmat(sources(i).data.cl.ch_c95, n_rows, 1);
end


writetable(cluster_coherence, 'v2cluster_coherence.csv');


%% Helper function

function pooled_coherence = calculate_cluster_pooled_coherence(coherence, config)
% CALCULATE_POOLED_COHERENCE - Pool coherence across specified participants within groups
%
% This function pools coherence data across specified participants and trials within each
% group for each force level. It takes the original coherence data from individual
% participants and combines it using the pool_scf and pool_scf_out functions from
% the NeuroSpec20 toolbox.
%
% Parameters:
%   coherence - Structure containing individual coherence results for each participant
%               Format: coherence.group_name.participant_name with fields:
%               - sc: complex coherence spectrum [frequency x channels x trials x force_levels]
%               - cl: coherence statistics structure [channels x trials x force_levels]
%   config - Analysis configuration parameters containing:
%            - FORCE_LEVELS: Cell array of force level strings (e.g., {'15', '35', '55', '70'})
%            - REPS: Cell array of repetition strings (e.g., {'1', '2', '3'})
%            - PARTICIPANTS: Structure specifying which participants to include:
%                           - dexterity: Cell array of participant names (e.g., {'P1', 'P2', 'P10'})
%                           - strength: Cell array of participant names (e.g., {'P3', 'P4', 'P5'})
%                           If not provided, all available participants will be used.
%
% Returns:
%   pooled_coherence - Structure containing pooled coherence results within groups
%                     Format: pooled_coherence.group_name.force_level with fields:
%                     - f: frequency vector
%                     - t: coherence estimates
%                     - cl: pooled coherence statistics
%                     - sc: pooled complex coherence spectrum
%
% Example:
%   config.PARTICIPANTS.dexterity = {'P1', 'P2', 'P10'};
%   config.PARTICIPANTS.strength = {'P3', 'P4', 'P5'};
%   pooled_coherence = calculate_pooled_coherence(coherence, config);
%   dexterity_15mvc = pooled_coherence.dexterity.MVC15;
%   strength_70mvc = pooled_coherence.strength.MVC70;

% Initialize output structure
pooled_coherence = struct('dexterity', struct(), 'strength', struct());

% Setup group and force level labels
group_names = {'dexterity', 'strength'};
force_level_labels = cellfun(@(x) sprintf('MVC%s', x), config.FORCE_LEVELS, 'UniformOutput', false);

% Pool coherence across participants within each group
for groupIdx = 1:length(group_names)
    group_name = group_names{groupIdx};
    
    % Get participants for this group
    if isfield(config, 'PARTICIPANTS') && isfield(config.PARTICIPANTS, group_name)
        % Use specified participants
        participants = config.PARTICIPANTS.(group_name);
        
        % Validate that specified participants exist in the coherence data
        available_participants = fieldnames(coherence.(group_name));
        missing_participants = setdiff(participants, available_participants);
        if ~isempty(missing_participants)
            warning('The following participants were specified but not found in %s group: %s', ...
                    group_name, strjoin(missing_participants, ', '));
        end
        
        % Filter to only include participants that actually exist
        participants = intersect(participants, available_participants);
    else
        % Use all available participants if none specified
        participants = fieldnames(coherence.(group_name));
    end
    
    if isempty(participants)
        warning('No valid participants found for %s group. Skipping...', group_name);
        continue;
    end

    % Process each force level
    for forceIdx = 1:length(config.FORCE_LEVELS)
        current_force_level = force_level_labels{forceIdx};
        
        % Print current processing information
        if groupIdx == 1
            fprintf('\nPooling Coherence across Participants for Dexterity-Trained at %sMVC:\n', config.FORCE_LEVELS{forceIdx});
        else
            fprintf('\nPooling Coherence across Participants for Strength-Trained at %sMVC:\n', config.FORCE_LEVELS{forceIdx});
        end
        
        % PROPERLY INITIALIZE pooling variables for each force level
        in_f = [];
        in_v = [];
        first_entry = true;
        
        % Pool across participants and trials
        for participantIdx = 1:length(participants)
            current_participant = participants{participantIdx};
            
            % Loop through each repetition/trial for this participant
            for trialIdx = 1:length(config.REPS)
                % Pool data - handle first entry separately
                if first_entry
                    [out_f, out_v] = pool_scf(coherence.(group_name).(current_participant).sc(:,:,trialIdx,forceIdx), ...
                                            coherence.(group_name).(current_participant).cl(:,:,trialIdx,forceIdx));
                    first_entry = false;
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

end