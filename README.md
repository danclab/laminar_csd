## HNN-lameg integration 

Where to find the original data: 
/home/common/bonaiuto/cued_action_meg/derivatives/
- processed
- processed_sss with motion correction
- processed_ss2 with motion correction 2nd version

## File description

**pipeline_00_localizer** is the first part: its role is to find the vertex with the peak activity in the precentral cortex during the close time window around the motor response
output: the time series at this peak vertex, written to a .txt file - serve as input for the **pipeline_01_fit_hnn**
*comments: could find other ways to select the peaks of interest, and possible problem with the clustering: check that does on local faces in roi and in downsampled data*

**pipeline_01_fit_hnn** we need as inputs the dipole file, that will be preprocessed with a 18Hz butteworth filter to make it easier for the HNN model to fit not to the noise
you can choose the model: law_2021 being the latest, the calcium_model with updated calcium dynamics, and jones_2009 as base 
allows you to optimize with bayesian optimizatio drive per drive, by starting with initial parameters and constraints on the parameters 
*comments: with too much trials (>50) or iteration (>100), crashes sometimes* 

**pipeline_02_extract_model_laminar_signals**
will use the optimized parameters written from pipeline_01_fit_HNN, rerun the model and then write the layer time series to .txt
will also compute the LFPs for the model and then CSD :to compare with the CSD from the lameg source reconstruction 
*comments: see model_merf.py comments, need to find a way to have the same distances and positionning of the model for comparison with the LFPs* 

**pipeline_03_synthetic_csd**
use the .txt layer (L2/3 and L5) time series that will be run through a forward model and generate synthetic sensor level data, then this will be run through the lameg reconstruction inverse_eeb
*comments: should merge with the pipeline_04 (that assess the significance of the laminar clusters, with permutation tests on single trials 
compute the RMSE between and plot with layer bounds*

**pipeline_04_synthetic_empirical_csd**
use the .txt layer (L2/3 and L5) time series that will be run through a forward model and generate synthetic sensor level data, then this will be run through the lameg reconstruction inverse_eeb + can specify several noise and version of the model 
comments: should merge with the pipeline_04 (that assess the significance of the laminar clusters, with permutation tests on single trials 
compute the RMSE between and plot with layer bounds

**pipeline_05_layer_dynamics_roi**
will compute the layer_dynamics for each time window (specified) of the erf in a region of interest (roi), baseline corrected then allow to vizualize this activity on the inflated surface. 
Second part is about the assessment of significance of clusters using single trials 
*comments: should correct the way the significance is tested, one is using the other with single trials time series: not optimal* 

### Utilities files
- **model_merf.py**
function that initialize the drives to the base HNN model, specifying the shanks as well for HNN-LFPs - with distances adapted to the motor cortex: could do this iteratively on extracting the thickness of the cortex at this vertex and put the L5 and L2/3 layer at the right distance (can change this in the model)
*comments: takes hypo as a variable: indeed, this is not the same pattern of drives*

- **single_trials_csd.py**
script that computes the CSD, extract the layer activity and compute absoute mean activity per time window and per layer specifified for single trials. Uses MPI shared processing and writing to not overload the RAM, as for each vertex(+11 layers) it extracts the single time series, then bin in subgroups to then compute the CSD on the binned trials (otherwise too heavy), extract and compute layer activity per bin per time window and write it to a dictionnary
take as input the MU matrix with the weights for source reconstruction computed on the averaged base data
*comments: the script may crash because of comm.Barrier() not being at the right place*

- **hnn_utils.py**
custom functions used for hnn, include the plotting of mean csd

- **utils.py**
custom functions for lameg: clustering, or compute_layer_activity...

## What needs to be done:
- clean the code
- run on all subjects the pipeline_05 mostly, then group level analysis of laminar activity clusters
- do an animation with the layer_activity in each time window
- do simulations at each layer combination at several vertices to see if different source reconstruction
- modify base model to take motor cortex characteristics