# ============ calculate_bootstrap_effectsize.R ============ #


# FUNCTION: apply_bootES

# Calculates Cohen's d effect size using bootES package

# Inputs:
# - dataframe: "demographic_anthropometric_force_data.csv". 
# - column_name: columns for analysis
# - number of bootstrap replicates: R = 10000 is used.

# Outputs:
# - BootES results: Cohen's d effect size and 95% confidence intervals, along with other metrics.

# Note: returns NULL for non-numeric columns

apply_bootES <- function(data, column_name, R = 10000) {
  if (is.numeric(data[[column_name]])) {
    # Prepare data for bootES analysis
    combined_boot <- data %>% 
      select(testing_group, all_of(column_name))
    
    # Run bootES with 10000 replicates
    bootES(combined_boot, 
           R, 
           data.col = column_name, 
           group.col = "testing_group",
           contrast = c("strength","dexterity"),
           effect.type = "cohens.d")
  } else {
    return(NULL)
  }
}



# FUNCTION: boot_results_to_df

# Converts bootES results to a df for easier viewing

# Inputs:
# - boot_list: a list of results from several colimns using the apply_bootES function.

# Outputs:
# - dataframe: containing Cohen's d effect size and 95% confidence intervals.

boot_results_to_df <- function(boot_list) {
  data.frame(
    Test_Variable = names(boot_list),
    Cohens_d = sapply(boot_list, function(x) round(x$t0, 5)),        # Round to 5 decimal places
    Lower_CI = sapply(boot_list, function(x) round(x$bounds[1], 5)), # Round to 5 decimal places
    Upper_CI = sapply(boot_list, function(x) round(x$bounds[2], 5))  # Round to 5 decimal places
  )
}
