# ============ force_plots_utils.R ============ #

# FUNCTION: create_force_plot

# Creates raincloud plots for maximum force metrics.

# Inputs:
# - data: "demographic_anthropometric_force_data.csv".
# - ci_data: output from the function "calculate_group_boot_confint" in "calculate_bootstrap_confidence_intervals.R".
# - force_var: column name of the force variable used (i.e. "max_force" or "max_force_forearm_circumference_normalised").
# - y_label: Label of the y-axis (such as "Force Output (N)")
# - strip label: 90 degree rotated plot title in grey shaded box (such as "Gross Force").

# Outputs:
# - Force variable raincloud plots.


create_force_plot <- function(data, ci_data, force_var, y_label, strip_label) {
  # Controls spacing of the rain/jitter plot elements
  rain_height <- 0.07
  
  # Add a dummy faceting variable to the data
  data$facet_var <- strip_label
  
  # Filter confidence interval data and add faceting variable
  ci_data_filtered <- ci_data[ci_data$Testing_Variable == force_var, ]
  ci_data_filtered$facet_var <- strip_label
  
  # Ensure consistent ordering of groups in visualization
  # 'strength' appears before 'dexterity' in the plot
  data$testing_group <- factor(data$testing_group, levels = c("strength", "dexterity"))
  
  # Begin building the plot with multiple layers.
  ggplot() +
    # Layer 1: Violin plots ("clouds").
    # Shows distribution kernel density of the data (uses the introdataviz package).
    introdataviz::geom_flat_violin(
      data = data, 
      aes(x = "", y = .data[[force_var]], fill = testing_group), 
      trim = FALSE,  # Don't trim the tails of the distribution
      alpha = 0.9,   # Slight transparency
      show.legend = FALSE,
      position = position_nudge(x = rain_height + 0.07)  # Shift position slightly
    ) +
    
    # Layer 2: Individual participant data points ("rain")
    # Jittered points showing participant raw data
    geom_point(
      data = data, 
      aes(x = "", y = .data[[force_var]], colour = testing_group, shape = testing_group), 
      size = 2.2,
      stroke = 0.5,
      show.legend = FALSE,  # Hide from legend since shown in violin plot
      position = position_jitter(width = rain_height - 0.055)  # Add random noise
    ) +
    
    # Layer 3: Box plots
    # Show quartiles and median
    geom_boxplot(
      data = data, 
      aes(x = "", y = .data[[force_var]], fill = testing_group), 
      width = 0.07, 
      show.legend = FALSE, # Hide from legend since shown in violin plot
      outlier.shape = NA,  # Hide outliers since shown in rain plot
      position = position_dodgenudge(width = 0.075, x = -rain_height * 1.8) # Change position to be below "rain".
    ) +
    
    # Layer 4: Confidence intervals
    # Show mean and CI from bootstrap analysis positioned inside the "clouds"
    geom_pointrange(
      data = ci_data_filtered, 
      aes(x = "", y = Mean, ymin = Lower_CI, ymax = Upper_CI, 
          color = testing_group, shape = testing_group, fill = testing_group), 
      linewidth = 1, 
      size = 0.6, 
      show.legend = FALSE,
      position = position_dodgenudge(x = rain_height + 0.14, 
                                     width = 0.05) # Position inside the "clouds".
    ) +
    
    # Add faceting with placeholder variable
    facet_grid(rows = vars(facet_var), switch = "y") +
    
    # Configure scales and coordinate system
    scale_x_discrete(name = "", expand = c(rain_height * 3, 0, 0, 0.7)) +
    scale_y_continuous() +
    coord_flip() +  # Flip coordinates for horizontal layout
    theme_bw() + 
    # Apply theme and styling which will be consistent across plots
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_blank(),
      panel.border = element_blank(),
      axis.line.y = element_blank(),
      axis.line.x = element_line(color = "grey70", linewidth = 0.5),
      axis.ticks.y = element_blank(),
      axis.ticks.x = element_line(color = "grey70", linewidth = 0.5),
      axis.title.y = element_blank(),
      axis.text.y = element_blank(),
      axis.title.x = element_text(face = "bold", size = 11),
      axis.text.x = element_text(size = 9),
      strip.background = element_rect(fill = "grey90", color = NA),  # Remove the box around facet labels
      legend.position = "none",
      strip.text.y = element_text(
        angle = 90,
        face = "bold",
        size = 12
      )
    ) +
    
    # Add labels and titles
    labs(
      y = y_label,
      fill = "Testing Group"
    ) +
    
    # Define consistent visual encoding for testing groups
    scale_shape_manual(values = c("strength" = 21, "dexterity" = 24),
                       labels = c("strength" = "Strength-Trained", "dexterity" = "Dexterity-Trained")) +
    scale_color_manual(values = c("strength" = "#fe9d5d", "dexterity" = "#365373"),
                       labels = c("strength" = "Strength-Trained", "dexterity" = "Dexterity-Trained")) +
    scale_fill_manual(values = c("strength" = "#fe9d5d", "dexterity" = "#456990"),
                      labels = c("strength" = "Strength-Trained", "dexterity" = "Dexterity-Trained"))
}


# FUNCTION: create_cov_force_plot

# Creates faceted plot that shows 95% CI and raw data points for foce steadiness (coefficient of varation of force)
# during the submaximal force trials

# Inputs:
# - data: "demographic_anthropometric_force_data.csv".
# - ci_data: output from the function "calculate_group_boot_confint" in "calculate_bootstrap_confidence_intervals.R".

# Outputs:
# - Force steadiness plot (coefficient of variation of force).

create_cov_force_plot <- function(data, ci_data) {
  # Prepare coefficient of variation (COV) of force data for plotting
  # Select relevant columns
  plot_cov <- data %>% 
    select(participant, testing_group, cov_force_15, cov_force_35, cov_force_55, cov_force_70)
  
  # Convert wide format to long format for plotting
  plot_cov_force_long <- plot_cov %>%
    pivot_longer(cols = -c(testing_group, participant), 
                 names_to = "Variable", 
                 values_to = "Value") %>% 
    # Recode variable names and set factor levels for proper ordering
    mutate(
      # Convert numeric suffixes to percentage labels
      Variable = factor(recode(Variable,
                               "cov_force_15" = "15%",
                               "cov_force_35" = "35%",
                               "cov_force_55" = "55%",
                               "cov_force_70" = "70%"),
                        levels = c("70%", "55%", "35%", "15%")),  # Order from high to low
      testing_group = factor(testing_group, levels = c("strength", "dexterity"))
    )
  
  # Similarly prepare confidence interval data
  plot_cov_force_CI <- ci_data %>% 
    filter(Testing_Variable %in% c("cov_force_15", "cov_force_35", "cov_force_55", "cov_force_70")) %>% 
    mutate(
      Variable = factor(recode(Testing_Variable,
                               "cov_force_15" = "15%",
                               "cov_force_35" = "35%",
                               "cov_force_55" = "55%",
                               "cov_force_70" = "70%"),
                        levels = c("70%", "55%", "35%", "15%")),
      testing_group = factor(testing_group, levels = c("strength", "dexterity"))
    )
  
  # Create multi-panel plot
  ggplot() + 
    # Layer 1: Individual participant data points with jitter.
    geom_jitter(data = plot_cov_force_long, 
                aes(x = Value, y = testing_group, color = testing_group, shape = testing_group), 
                size = 2.5, 
                stroke = 0.5, 
                position = position_jitterdodge(jitter.width = 0.05, dodge.width = 0.5),
                show.legend = FALSE) +
    
    # Layer 2: Bootstrapped mean and 95% confidence intervals
    geom_pointrange(data = plot_cov_force_CI, 
                    aes(x = Mean, y = testing_group, 
                        xmin = Lower_CI, xmax = Upper_CI, 
                        color = testing_group, shape = testing_group, 
                        fill = testing_group),
                    size = 1, 
                    linewidth = 1.7, 
                    position = position_dodge(width = 0.5)) +
    
    # Create separate panels for each force level
    facet_grid(rows = vars(Variable), scales = "free_y", switch = "y") + 
    
    # Apply theme and styling which will be consistent across plots
    theme_bw() +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(colour = "grey70"),
      legend.position = "bottom",
      legend.title = element_text(size = 9, face = "bold"),
      legend.background = element_blank(),
      legend.margin = margin(t = -7.5), # Adjust legend position (vertically up)
      legend.text = element_text(size = 9),
      axis.title.y = element_text(angle = 90, size = 12, face = "bold", hjust = 0.5, 
                                  vjust =0.5, margin = margin(r = 10)),
      axis.text.y = element_blank(), 
      axis.ticks.y = element_blank(),
      axis.line.y = element_blank(),
      strip.text.y = element_text(angle = 90, size = 11, face = "bold"),
      axis.title.x = element_text(size = 11, face = "bold", margin = margin(t = 15)),
      axis.text.x = element_text(size = 10, face = "bold"),
      axis.line.x = element_line(size = 0.3, colour = "grey80"),
      title = element_blank(),
      strip.background = element_rect(fill = "grey90", color = NA)
    ) +
    
    # Add labels and configure scales
    labs(x = "COV",
         y = "Percentage of MVC",
         fill = "Testing Group",
         color = "Testing Group", 
         shape = "Testing Group") +
    scale_x_continuous(labels = scales::percent) +  # Format x-axis as force level percentages
    
    # Apply consistent visual encoding for testing groups
    scale_shape_manual(values = c("strength" = 21, "dexterity" = 24),
                       labels = c("strength" = "Strength-Trained", "dexterity" = "Dexterity-Trained")) +
    scale_color_manual(values = c("strength" = "#fe9d5d", "dexterity" = "#456990"),
                       labels = c("strength" = "Strength-Trained", "dexterity" = "Dexterity-Trained")) +
    scale_fill_manual(values = c("strength" = "#fe9d5d", "dexterity" = "#456990"),
                      labels = c("strength" = "Strength-Trained", "dexterity" = "Dexterity-Trained"))
}