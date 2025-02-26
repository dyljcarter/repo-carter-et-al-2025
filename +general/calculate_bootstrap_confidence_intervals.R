# ============ calculate_bootstrap_confidence_intervals.R ============ #


#-----------------#
# MAIN FUNCTIONS
#-----------------#


# FUNCTION: calculate_group_boot_confint

# Calculates bootstrap confidence intervals separately for dexterity and strength groups

# Inputs:
# - dataframe: "demographic_anthropometric_force_data.csv". 
# - columns to exclude: participant ID and testing group (as these do not need to be analysed).
# - number of bootstrap replicates: R = 10000 is used.

# Outputs:
# - dataframe containing bootstrapped means and 95% confidence intervals for each group/column

calculate_group_boot_confint <- function(data, exclude_columns = c("participant", "testing_group"), R = 10000) {
  # Get columns to analyze (excluding specified ones)
  columns_to_include <- setdiff(names(data), exclude_columns)
  
  # Process each column with the bootstrapping "boot" function.
  ci_means <- lapply(columns_to_include, function(column) {
    # Bootstrap for dexterity group means
    results_dexterity <- boot(data, 
                              \(data, indices) mean_per_group(data, indices, "dexterity", column), 
                              R)
    # Bootstrap for strength group means
    results_strength <- boot(data, 
                             \(data, indices) mean_per_group(data, indices, "strength", column), 
                             R)
    
    # Calculate 95% confidence intervals using percentile method
    ci_dexterity <- boot.ci(results_dexterity, type = "perc")
    ci_strength <- boot.ci(results_strength, type = "perc")
    
    # Combine results into single dataframe with group means and 95% CIs
    data.frame(
      Testing_Variable = column,
      testing_group = c("dexterity", "strength"),
      Mean = c(mean(data[[column]][data$testing_group == "dexterity"]),
               mean(data[[column]][data$testing_group == "strength"])),
      Lower_CI = c(ci_dexterity$percent[4], ci_strength$percent[4]),  # 2.5th percentile
      Upper_CI = c(ci_dexterity$percent[5], ci_strength$percent[5])   # 97.5th percentile
    )
  })
  do.call(rbind, ci_means) %>%   # Combine all results
    mutate(across(c('Mean', 'Lower_CI', 'Upper_CI'), \(x) round(x, 5))) # Round to 5 decimal places
}


# FUNCTION: calculate_meandiff_boot_confint

# Calculates bootstrap confidence intervals for differences between groups

# Inputs:
# - dataframe: "demographic_anthropometric_force_data.csv". 
# - columns to exclude: participant ID and testing group (as these do not need to be analysed).
# - number of bootstrap replicates: R = 10000 is used.

# Outputs:
# - dataframe containing bootstrapped group difference of means and their 95% confidence intervals.


calculate_meandiff_boot_confint <- function(data, exclude_columns = c("participant", "testing_group"), R = 10000) {
  columns_to_include <- setdiff(names(data), exclude_columns)
  
  # Process each column with the bootstrapping "boot" function.
  ci_results <- lapply(columns_to_include, function(column) {
    # Calculate bootstrap samples of mean differences
    results <- boot(data, \(data, indices) mean_diff(data, indices, column), R)
    ci <- boot.ci(results, type = "perc")  # Get confidence intervals
    
    # Store results with column name
    data.frame(
      Testing_Variable = column,
      Mean_Diff = mean_diff(data, 1:nrow(data), column),
      Lower_CI = ci$percent[4],    # 2.5th percentile
      Upper_CI = ci$percent[5]     # 97.5th percentile
    )
  })
  # Combine and round results
  do.call(rbind, ci_results) %>% 
    mutate(across(c('Mean_Diff', 'Lower_CI', 'Upper_CI'), \(x) round(x, 5))) # Round to 5 decimal places
}



#-----------------#
# HELPER FUNCTIONS
#-----------------#


# HELPER FUNCTION: mean_per_group

# Helper function to calculate mean for a specific group during bootstrap
# Used by calculate_group_boot_confint

mean_per_group <- function(data, indices, testing_group, column) {
  d <- data[indices, ]  # Get bootstrap sample
  group_data <- d[[column]][d$testing_group == testing_group]  # Extract group data
  return(mean(group_data))  # Calculate mean
}


# HELPER FUNCTION: mean_diff

# Helper function to calculate difference in means between groups
# Used by calculate_meandiff_boot_confint

mean_diff <- function(data, indices, column) {
  d <- data[indices, ]  # Get bootstrap sample
  # Calculate means for each group
  dexterity <- d[[column]][d$testing_group == "dexterity"]
  strength <- d[[column]][d$testing_group == "strength"]
  return(mean(dexterity) - mean(strength))  # Return difference
}
