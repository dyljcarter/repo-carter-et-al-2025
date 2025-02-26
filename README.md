# REPO-CARTER-ET-AL-2025
## Overview
This repository contains all code used to analyse data for the manuscript:
"Neuromuscular strategies for regulating hand force are influenced by training history"  
by Dylan J. Carter, James R. Forsyth, Joshua P.M. Mattock and Jonathan Shemmell
## Analysis Components

### 1. Intermuscular Coherence Analysis
#### MATLAB Data Analysis
`+coherence/+data_analysis/coherence_example_analysis.mlx`


> *Dependencies:*
> - *+coherence/+data_analysis/calculate_coherence.m*
> - *+coherence/+data_analysis/neurospec_plot_coherence.m*
> - *+coherence/+data_analysis/plotting_data_for_R.m*
> - *NeuroSpec 2.0 Toolbox (by David Halliday) - [Download here](https://github.com/dmhalliday/NeuroSpec/blob/master/neurospec20.zip)*
#### R Plotting
`+coherence/+plotting/final_coherence_plots.qmd`

> *Dependencies:*
> - *+coherence/+plotting/coherence_plot_utils.R*

### 2. Motor Unit Activity Analysis
#### MATLAB Data Analysis
`+mu/+data_analysis/mu_example_analysis.mlx`

> *Dependencies:*
> - *+mu/+data_analysis/+process_mu_data.m*
#### R Statistical Analysis
`+mu/+statisical_analysis/mu_statistical_analysis.qmd`

> *Dependencies:*
> - *+mu/+statistical_analysis/mu_plots_utils.R*

### 3. General Data Analysis
Covers anthropometric, demographic, and force data analysis.
#### R Statistical Analysis
`+general/general_analysis.qmd`

> *Dependencies:*
> - *+general/calculate_bootsrap_confidence_intervals.R*
> - *+general/calculate_bootsrap_effectsize.R*
> - *+general/descriptives_table_util.R*
> - *+general/force_plots_utils.R*

## Analysis Output
Click the links below or download the repository to view these HTML files:
- Coherence: [+coherence/+plotting/final_coherence_plots.html](https://htmlpreview.github.io/?https://github.com/dyljcarter/repo-carter-et-al-2025/blob/main/%2Bcoherence/%2Bplotting/final_coherence_plots.html)
- Motor Unit: [+mu/+statistical_analysis/mu_statistical_analysis.html](https://htmlpreview.github.io/?https://github.com/dyljcarter/repo-carter-et-al-2025/blob/main/%2Bmu/%2Bstatistical_analysis/mu_statistical_analysis.html)
- General: [+general/general_analysis.html](https://htmlpreview.github.io/?https://github.com/dyljcarter/repo-carter-et-al-2025/blob/main/%2Bgeneral/general_analysis.html)

## Data Availability
For data files to rerun this analysis, please contact: dcarter@uow.edu.au
## Authors
### Manuscript Authors
- Dylan J. Carter¹,²
- James R. Forsyth²
- Joshua P.M. Mattock²
- Jonathan Shemmell¹
### Analysis Code Author
- Dylan J. Carter¹,²
### Affiliations
1. **Neuromotor Adaptation Laboratory**
   
   School of Medical, Indigenous, and Health Sciences  
   Faculty of Science, Medicine, and Health  
   University of Wollongong
2. **Biomechanics Research Laboratory** 

   School of Medical, Indigenous, and Health Sciences  
   Faculty of Science, Medicine, and Health  
   University of Wollongong
## Acknowledgments
- Analysis uses the NeuroSpec 2.0 toolbox by Dr David Halliday et al. ([GitHub](https://github.com/dmhalliday/NeuroSpec))
- Thank you to Dr David Halliday and Dr Tjeerd Boonstra for their assistance with intermuscular coherence analysis implementation
