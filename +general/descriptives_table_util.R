# ============ descriptives_table_utils.R ============ #

# FUNCTION: create_descriptives_table

# Creates formatted table for several descriptive variables.
# (age, height, mass, hand training time, testing group)

# Inputs:
# - data: "demographic_anthropometric_force_data.csv".

# Outputs:
# - Formatted table of descriptive measures

create_descriptives_table <- function(data) {
  # Step 1: Select and rename relevant columns for analysis
  # Select demographic variables and testing group
  # Rename columns to more readable format for final table
  general_descriptives <- data %>%
    select(age, height, mass, specific_training_time, testing_group) %>%
    rename(
      `Hand Training Time` = specific_training_time,
      `Age` = age,
      `Height` = height,
      `Mass` = mass
    ) %>%
    
    # Step 2: Calculate summary statistics by testing group
    # Group data by testing_group and calculate mean and SD for each variable
    # .names parameter creates new column names in format: variable_statistic
    group_by(testing_group) %>%
    summarise(across(everything(), list(
      mean = ~mean(., na.rm = TRUE),  # Calculate mean, removing NA values
      sd = ~sd(., na.rm = TRUE)       # Calculate std dev, removing NA values
    ), .names = "{col}_{fn}")) %>%
    
    # Step 3: Reshape data for presentation
    # First pivot: Transform summary statistics into separate rows
    # Converts from wide to long format
    pivot_longer(
      cols = -testing_group,           # Keep testing_group as is
      names_to = c("Variable", ".value"), # Split column names into variable name and statistic type
      names_pattern = "^(.*)_(mean|sd)$"  # Pattern to match column names
    ) %>%
    
    # Second pivot: Create separate columns for each testing group
    # Transforms data to have mean and SD columns for each group
    pivot_wider(
      names_from = testing_group,      # Create columns based on testing group
      values_from = c(mean, sd)        # Spread mean and SD values
    )
  
  # Step 4: Format table using gt (great table) package
  general_descriptives %>%
    gt() %>%
    # Format all numeric columns to 2 decimal places
    fmt_number(
      columns = everything(),
      decimals = 2
    ) %>%
    
    # Step 5: Rename columns with clear, formatted labels
    cols_label(
      Variable = "Measure",
      mean_dexterity = "Mean",
      sd_dexterity = md("&plusmn;SD"),  # Plus/minus symbol (markdown)
      mean_strength = "Mean",
      sd_strength = md("&plusmn;SD")    # Plus/minus symbol (markdown)
    ) %>%
    
    # Step 6: Add column groupings (spanners)
    # Group strength-related columns
    tab_spanner(
      label = "Strength-trained",
      columns = c(mean_strength, sd_strength)
    ) %>%
    # Group dexterity-related columns
    tab_spanner(
      label = "Dexterity-Trained",
      columns = c(mean_dexterity, sd_dexterity)
    ) %>%
    
    # Step 7: Set column widths for consistent formatting
    cols_width(
      Variable ~ px(250),       # Variable names get more space
      everything() ~ px(100)    # All other columns same width
    ) %>% 
    
    # Step 8: Apply table styling options
    tab_options(
      table.background.color = "white",
      row.striping.background_color = "white",
      table_body.hlines.style = "none",           # Remove horizontal lines in body
      column_labels.border.bottom.color = "black", # Add borders to column labels
      column_labels.border.top.color = "black",
      table_body.border.bottom.color = "black",   # Add border to bottom of table
      table.border.top.color = "black",           # Add border to top of table
      table.border.bottom.color = "black",
      heading.border.bottom.color = "black"
    ) %>% 
    
    # Step 9: Disable row striping
    opt_row_striping(
      row_striping = FALSE
    )
}