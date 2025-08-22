# REPO-CARTER-ET-AL-2025
## Overview
This repository contains all code used to analyse data for the manuscript:
"Neuromuscular control of a five-finger pinch task is influenced by training history."  
by Dylan J. Carter, James R. Forsyth, Joshua P.M. Mattock and Jonathan Shemmell (2025)
Published in Experimental Brain Research.

## Statistical analysis (R)
`statistical_analysis.qmd`

### Output from statistical analysis
Click the links below or download the repository to view these HTML files:
- Coherence: [statistical_analysis.html](https://htmlpreview.github.io/?https://github.com/dyljcarter/repo-carter-et-al-2025/blob/main/%2Bcoherence/%2Bplotting/final_coherence_plots.html)


## Data analysis (MATLAB)

### 1. Intermuscular coherence analysis
`+coherence/calculate_coherence.m`
`+coherence/export_coherenceData_for_R.m`

> *Dependencies:*
> - *NeuroSpec 2.0 Toolbox (by David Halliday) - [Download here](https://github.com/dmhalliday/NeuroSpec/blob/master/neurospec20.zip)*

### 2. Motor unit activity analysis
`+mu/+process_mu_data.m`

### 3. Clustering analysis
`+rms/calculate_EMGrms.m`

### 5. Online Resource 1 analysis
`+coherence/calculate_cluster_coherence.m`

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
