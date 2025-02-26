# ============ mu_plots_utils.R ============ #

# FUNCTION: create_mu_beeswarm

# Creates split beeswarm plots for motor unit thresholds at each force level and for each muscle.

# Inputs:
# - data: output table from the matlab function "process_mu_data.m" - "analysed_mu_data.csv".

# Outputs:
# - Split beeswarm plot for motor unit thresholds faceted by force level and muscle.

create_mu_beeswarm <- function(data) {
  
  # Transform input data:
  # - Convert muscle labels to uppercase
  # - Rename testing groups to more readable format
  data <- data %>%
    mutate(
      muscle = toupper(muscle),  # Convert APB/FDS to uppercase
      testing_group = case_when(  # Rename groups for better readability
        testing_group == "strength" ~ "Strength-Trained",
        testing_group == "dexterity" ~ "Dexterity-Trained",
        TRUE ~ testing_group
      )
    )
  
  # Calculate sample size for each combination of:
  # - force level (15%, 35%, 55% and 70%)
  # - muscle (APB, FDS)
  # - testing group (Strength, Dexterity)
  # This will be displayed below each group in the plot
  count_data <- data %>%
    group_by(force_level, muscle, testing_group) %>%
    summarise(count = n(), .groups = 'drop') %>%
    mutate(y_pos = -3)  # Position for count labels below the data points
  
  # Create the base beeswarm plot
  plot <- ggplot(data) + 
    # Add jittered points (beeswarm style) for each data point
    geom_quasirandom(
      aes(
        x = muscle,  # X-axis shows muscle type (APB/FDS)
        y = firing_threshold,  # Y-axis shows motor unit recruitment threshold
        group = testing_group,  # Separate points by training group
        colour = testing_group,  # Color points by training group
        shape = testing_group,  # Different shapes for each group
        fill = testing_group  # Fill color for shapes
      ),
      width = 0.3,  # Controls horizontal spread of points
      size = 1    # Size of individual points
    ) +
    # Add sample size labels below each group
    geom_text(
      data = count_data,
      aes(x = muscle, y = y_pos, 
          label = count,
          color = testing_group,
          group = testing_group),
      show.legend = FALSE,
      position = position_dodge(width = 0.6),
      size = 8/.pt,
      vjust = 1
    ) +
    geom_hline(yintercept = 0, linewidth = 0.2, colour = "grey80") +
    # Create separate panels for each force level
    facet_wrap(vars(force_level), nrow = 1, 
               labeller = labeller(force_level = function(x) paste0(x, "%"))) +
    # Add axis labels and legend titles
    labs(x = "Muscle", y = "Recruitment Threshold\n(% of MVC)", color = "Testing Group", 
         shape = "Testing Group", fill = "Testing Group") +
    # Use black and white theme as base
    # Apply theme and styling which will be consistent across plots
    theme_bw() +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(colour = "grey70"),
      legend.position = "none",
      strip.text = element_text(size = 11, face = "bold"),
      axis.title = element_text(size = 11, face = "bold"),
      axis.text.x = element_text(size = 9, face = "bold"),
      axis.line.x = element_line(size = 0.3, colour = "grey80"),
      axis.text.y = element_text(size = 9),
      title = element_blank(),
      strip.background = element_rect(fill = "grey90", color = NA) 
    ) +
    # Set custom shapes, colors and fills for groups
    scale_shape_manual(values = c("Strength-Trained" = 21, "Dexterity-Trained" = 24)) +
    scale_color_manual(values = c("Strength-Trained" = "#fe9d5d", "Dexterity-Trained" = "#456990")) +
    scale_fill_manual(values = c("Strength-Trained" = "#fe9d5d", "Dexterity-Trained" = "#456990")) +
    
    # Configure y-axis scale
    scale_y_continuous(
      breaks = c(0, 25, 50, 75), # Scale for each force level
      labels = c("0%", "25%", "50%", "75%"),
      limits = c(-5, 75)  # Include space for count labels at bottom
    )
  
  # Modify point positions to create split beeswarm effect
  # This separates strength and dexterity groups horizontally
  p <- ggplot_build(plot)
  p$data[[1]] <- p$data[[1]] %>%
    mutate(
      diff = abs(x-round(x)+0.1),  # Calculate offset from center
      x = case_when(  # Apply offset in opposite directions for each group
        group %% 2 == 0 ~ round(x) + diff,
        TRUE ~ round(x) - diff
      )
    ) %>%
    dplyr::select(-diff)
  
  # Convert back to gtable and return
  ggplot_gtable(p)
}



# FUNCTION: create_emmeans_plot

# Creates geom_point plot for the estimated marginal mean motor unit firing rate 
# at each force level and for each muscle.

# Inputs:
# - data: output from the EMmeans function used for calculation of estimated marginal means            in linear
#         mixed effects models

# Outputs:
# - Plot of EMM average motor unit firing rate faceted by muscle.


create_emmeans_plot <- function(emm) {
  
  # Process estimated marginal means data:
  # - Rename columns for clarity 
  # - Convert force levels to numeric
  # - Format group names
  # - Filter to include only specific force levels for each muscle
  processed_emm_data <- as.data.frame(emm$emmeans) %>%
    rename(Mean = emmean, 
           Lower_CI = lower.CL, 
           Upper_CI = upper.CL) %>%
    mutate(
      force_level = as.numeric(gsub("MVC", "", as.character(force_level))),
      muscle = case_when(
        muscle == "apb" ~ "Abductor Pollicis Brevis",
        muscle == "fds" ~ "Flexor Digitorum Superficialis",
        TRUE ~ muscle
      ),
      testing_group = case_when(
        testing_group == "strength" ~ "Strength-Trained",
        testing_group == "dexterity" ~ "Dexterity-Trained",
        TRUE ~ testing_group
      )
    ) %>% 
    # Filter data to include:
    # - APB muscle: 15%, 35%, 55%, 70% force levels
    # - FDS muscle: 35%, 55%, 70% force levels
    filter((muscle == "Abductor Pollicis Brevis" & force_level %in% c(15, 35, 55, 70)) |
             (muscle == "Flexor Digitorum Superficialis" & force_level %in% c(35, 55, 70)))
  
  # Create plot showing estimated means and confidence intervals
  ggplot(processed_emm_data, aes(x = force_level, color = testing_group)) +
    # Add fine lines connecting means within each group
    geom_line(aes(y = Mean, group = testing_group, linetype = testing_group), 
              linewidth = 0.2, position = position_dodge(width = 2)) +
    # Add points with error bars showing confidence intervals
    geom_pointrange(aes(y = Mean, ymin = Lower_CI, ymax = Upper_CI, 
                        shape = testing_group, fill = testing_group), 
                    linewidth = 1.5, size = 0.7, position = position_dodge(width = 2)) +
    # Add labels and titles
    labs(
      x = "Percentage of MVC",
      y = "Estimated Marginal Mean\nFiring Rate",
      color = "Testing Group",
      shape = "Testing Group",
      fill = "Testing Group",
      linetype = "Testing Group"
    ) +
    # Apply black and white theme and customize appearance
    theme_bw() +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(colour = "grey70"),
      legend.position = "bottom",
      strip.text = element_text(size = 12, face = "bold"),
      axis.title.x = element_text(size = 11, face = "bold", margin = margin(t = 15)),
      axis.text.x = element_text(size = 9, face = "bold"),
      axis.line.x = element_line(size = 0.3, colour = "grey80"),
      axis.title.y.left = element_text(size = 11, face = "bold", margin = margin(r = 10)),
      axis.text.y = element_text(size = 9),
      title = element_blank(),
      strip.background = element_rect(fill = "grey90", color = NA) ,
      legend.title = element_text(size = 9, face = "bold"),
      legend.text = element_text(size = 9)
    ) +
    # Configure x-axis scale and labels
    scale_x_continuous(
      limits = c(10, 75),
      labels = scales::percent_format(scale = 1),
      breaks = c(15, 35, 55, 70)
    ) +
    # Set custom shapes, colors, fills and line types for groups
    scale_shape_manual(values = c("Strength-Trained" = 21, "Dexterity-Trained" = 24)) +
    scale_color_manual(values = c("Strength-Trained" = "#fe9d5d", "Dexterity-Trained" = "#456990")) +
    scale_fill_manual(values = c("Strength-Trained" = "#fe9d5d", "Dexterity-Trained" = "#456990")) +
    scale_linetype_manual(values = c("Strength-Trained" = 5, "Dexterity-Trained" = 1)) +
    # Create separate panels for each muscle
    facet_wrap(~ muscle)
}