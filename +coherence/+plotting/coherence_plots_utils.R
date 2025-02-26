# ============ coherence_plots_utils.R ============ #

# FUNCTION: create_pooled_coherence_plots

# Creates NeuroSpec-style plots for pooled coherence estimates.

# Inputs:
# - data: "pooled_coherence_data.csv".
# - testing_group: either "strength" or "dexterity" for testing group.
# - force_level: 15, 35, 55 or 70. This is the MVC level for each submaximal test.

# Outputs:
# - Pooled coherence plot for a given group and force level.

create_pooled_coherence_plots <- function(data, testing_group, force_level) {
  # Construct column names dynamically
  coh_col <- paste0(testing_group, "_coh_", force_level)
  ci_col <- paste0(testing_group, "_ci_", force_level)
  # Create the plot using modern tidy evaluation
  p <- ggplot(data, aes(x = freq)) +
    # Add shaded frequency bands
    annotate("rect", xmin = 5, xmax = 15, ymin = 0, ymax = 0.035,
             fill = "grey90", alpha = 0.5) +
    annotate("rect", xmin = 15, xmax = 30, ymin = 0, ymax = 0.035,
             fill = "grey75", alpha = 0.5) +
    annotate("rect", xmin = 30, xmax = 60, ymin = 0, ymax = 0.035,
             fill = "grey60", alpha = 0.5) +
    # Add symbols
    annotate("text", x = 10, y = 0.03, label = "\u03B1", size = 12/.pt) +
    annotate("text", x = 22.5, y = 0.03, label = "\u03B2", size = 12/.pt) +
    annotate("text", x = 45, y = 0.03, label = "\u03B3", size = 12/.pt) +
    # Add coherence line with color
    geom_line(aes(y = .data[[coh_col]], color = testing_group), linewidth = 0.65) +
    # Add confidence interval as dashed line
    geom_hline(yintercept = data[[ci_col]][1], linetype = "dashed", linewidth = 0.4) +
    # Set axis limits and breaks with explicit names
    scale_x_continuous(
      limits = c(0, 65),
      name = "Frequency (Hz)",
      expand = expansion(mult = c(0, 0.05))  # This removes the gap at the bottom
    ) +
    scale_y_continuous(
      limits = c(0, 0.035), 
      breaks = c(0, 0.01, 0.02, 0.03),
      name = "Coherence Estimate",
      expand = expansion(mult = c(0, 0.05))  # This removes the gap at the bottom
    ) +
    # Add color scale
    scale_color_manual(
      values = c("strength" = "#fe9d5d", "dexterity" = "#456990"),
      labels = c("strength" = "Strength-Trained", "dexterity" = "Dexterity-Trained")
    ) +
    # Title only in labs (no x/y labels here)
    labs(title = paste0(force_level, "% MVC")) +
    # Modified theme
    theme_bw() +
    theme(
      panel.grid = element_blank(),
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      legend.position = "none",
      panel.border = element_blank(),
      axis.line.x = element_line(linewidth = 0.25, linetype = "solid", colour = "black"),
      axis.line.y = element_line(linewidth = 0.25, linetype = "solid", colour = "black"),
      axis.title = element_text(size = 11, face = "bold"),
      axis.text = element_text(size = 9)
    )
  return(p)
}


# FUNCTION: create_comparison_of_coherence_plots

# Creates NeuroSpec-style plots for comparison of coherence

# Inputs:
# - data: "comparison_of_coherence_data.csv".
# - force_level: 15, 35, 55 or 70. This is the MVC level for each submaximal test.
# - ci_label_x: x-dimension position of the "testing group" title for each 95% CI (allows repositioning).
# - band_label_y: y-dimension position of the "coherence band" title for the symbols within shaded areas
#                 (allows repositioning).

# Outputs:
# - Comparison of coherence plot for a given force level.


plot_comparison_of_coherence <- function(data, force_level, ci_label_x = 63, band_label_y = 0.16) {
  # Validate inputs
  coherence_col <- paste0("compcoh_", force_level)
  ci_col <- paste0("ci_", force_level)
  
  # Get CI value from data (taking first value since they're all the same)
  ci_value <- data[[ci_col]][1]
  
  # Create the plot using modern tidy evaluation
  p <- ggplot(data, aes(x = freq)) +
    # Add shaded frequency bands
    annotate("rect", xmin = 8, xmax = 16, ymin = -ci_value, ymax = ci_value,
             fill = "grey90", alpha = 0.5) +
    annotate("rect", xmin = 16, xmax = 30, ymin = -ci_value, ymax = ci_value,
             fill = "grey75", alpha = 0.5) +
    annotate("rect", xmin = 30, xmax = 60, ymin = -ci_value, ymax = ci_value,
             fill = "grey60", alpha = 0.5) +
    # Add symbols for frequency bands
    annotate("text", x = 12, y = band_label_y, label = "\u03B1", size = 12/.pt) +
    annotate("text", x = 22, y = band_label_y, label = "\u03B2", size = 12/.pt) +
    annotate("text", x = 41, y = band_label_y, label = "\u03B3", size = 12/.pt) +
    # Add ci line labels
    annotate("text", x = ci_label_x, y = 0.145, label = "Strength-trained", size = 8/.pt) +
    annotate("text", x = ci_label_x, y = -0.14, label = "Dexterity-trained", size = 8/.pt) +
    # Add confidence intervals as horizontal lines
    geom_hline(yintercept = ci_value, color = "#fe9d5d") +
    geom_hline(yintercept = -ci_value, color = "#456990") +
    # Add zero line
    geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.3) +
    # Add coherence line
    geom_line(aes(y = .data[[coherence_col]]), linewidth = 0.5) +
    # Set axis limits and breaks
    scale_x_continuous(
      limits = c(0, 60),
      breaks = seq(0, 65, by = 10),
      name = "Frequency (Hz)",
      expand = expansion(mult = c(0, 0.05))  # This removes the gap at the bottom
    ) +
    scale_y_continuous(
      limits = c(-0.23, 0.24),
      name = "Difference of Coherence",
      expand = expansion(mult = c(0, 0.05))  # This removes the gap at the bottom
    ) +
    # Title only in labs (no x/y labels here)
    labs(title = paste0(force_level, "% MVC")) +
    # Modified theme
    theme_bw() +
    theme(
      panel.grid = element_blank(),
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      legend.position = "none",
      panel.border = element_blank(),
      axis.line.x = element_line(linewidth = 0.25, linetype = "solid", colour = "black"),
      axis.line.y = element_line(linewidth = 0.25, linetype = "solid", colour = "black"),
      axis.title = element_text(size = 11, face = "bold"),
      axis.text = element_text(size = 9)
    )
  
  return(p)
}


# FUNCTION: compare_with_ci

# Creates tables of significant frequencies based on the coherence and comparison of coherence
# confidence limits and confidence intervals from the Neurospec Toolbox.

# Inputs:
# - data: "pooled_coherence_data" or comparison_of_coherence_data.csv".
# - freq_limit: Limit for frequencies of interest. Here = 60, as this is the end of the Gamma Band.
# - ci: if TRUE, will allow for confidence intervals (ie values negative but >CI) are still classified 
#       as significant. Default = FALSE.


# Outputs:
# - Table of significant frequencies.


compare_with_ci <- function(df, freq_limit = 60, ci = FALSE) {
  # Get the column names for comparison (excluding freq column)
  data_cols <- names(df)[2:ncol(df)]
  
  # Create pairs of columns to compare
  col_pairs <- matrix(data_cols, ncol = 2, byrow = TRUE)
  
  # Initialize empty lists for results
  all_mvcs <- character()
  all_groups <- character()
  all_frequencies <- numeric()
  
  # Process each pair of columns
  for(i in 1:nrow(col_pairs)) {
    col1 <- col_pairs[i, 1]
    col2 <- col_pairs[i, 2]
    
    if(ci) {
      # Extract MVC value from column name (assuming format like "compcoh_15")
      mvc_value <- str_extract(col1, "\\d+")
      
      # For CI=TRUE, compare absolute values and determine group based on sign
      comparison_df <- df %>%
        filter(freq <= freq_limit) %>%
        filter(abs(.data[[col1]]) > abs(.data[[col2]])) %>%
        mutate(
          group = case_when(
            .data[[col1]] > 0 ~ "Strength",  # Positive values indicate Strength
            .data[[col1]] < 0 ~ "Dexterity", # Negative values indicate Dexterity
            TRUE ~ NA_character_
          )
        )
      
      # Add results to vectors
      all_mvcs <- c(all_mvcs, rep(mvc_value, nrow(comparison_df)))
      all_groups <- c(all_groups, comparison_df$group)
      all_frequencies <- c(all_frequencies, comparison_df$freq)
      
    } else {
      # For CI=FALSE, extract MVC value from column name (assuming format like "strength_coh_15")
      mvc_value <- str_extract(col1, "\\d+")
      
      # Compare direct values
      comparison_df <- df %>%
        filter(freq <= freq_limit, .data[[col1]] > .data[[col2]])
      
      # Determine group from column name
      group <- if(grepl("strength", col1, ignore.case = TRUE)) "Strength" else "Dexterity"
      
      # Add results to vectors
      all_mvcs <- c(all_mvcs, rep(mvc_value, nrow(comparison_df)))
      all_groups <- c(all_groups, rep(group, nrow(comparison_df)))
      all_frequencies <- c(all_frequencies, comparison_df$freq)
    }
  }
  
  # Create final long format dataframe
  results_df <- data.frame(
    MVC = all_mvcs,
    Group = all_groups,
    Frequencies = all_frequencies
  )
  
  # Convert MVC to numeric and sort
  results_df$MVC <- as.numeric(results_df$MVC)
  results_df <- results_df %>%
    arrange(MVC, Group)
  
  return(results_df)
}