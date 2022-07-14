# COVID-19

## Detection of COVID-19 from chest CT 
The project addresses the issue of detecting COVID-19 infections by analysing chest 
CT scans using 3D Inception-based models to distinguish among patients with no
infections, affected by common pneumonia, and affected by COVID-19 pneumonia. 
To this end, first a set of individual models is built starting from 3D Inception-V1 and
Inception-V3 CNNs, then a soft-voting ensemble approach has been experimented to improve
their performance. 

## Dataset
The study is based on the CLEAN-CC-CCII dataset proposed by He et al. (2020). CLEA-CC-CCII is a cleaned version of the China Consortium 
of Chest CT Image Investigation (CC-CCII) dataset [2]. The dataset consists of chest CT scans obtained from patients without lung infections 
(Normal), with novel corona virus pneumonia (NCP) and common pneumonia (CP). 
The CLEAN-CC-CCII dataset is available at the following URL address: https://github.com/HKBU-HPML/HKBU_HPML_COVID-19 .

### Pre-processing
In the CLEAN-CC-CCII scans are released as .png files each representing a slice. Initially, the slices of each scans have been combined 
to convert them into NIFTI files. Morevoer, the CLEAN-CC-CCII consists of CT scans with slices of 512x512 pixels and a non-uniform depth size. 
Therefore, the scans have been pre-processed so that each of them has the same depth size. 
With the aim to assess how the scan depth affects the classifier’s performance the CT scans were processed to generate two sets of scans 
with two distinct depth size (i.e, 25 and 35 slices). These sets are built down-sampling or over-sampling the CT scans at the ends.
Please, use the Python notebook labeled "pre-processing" to perform these tasks.

## Testing
A 5-fold cross-validation strategy is used to test the models. Due to this testing strategy, at each step the dataset is automatically 
split into training (80%) and test (20%) –also taking care of maintaining the original imbalance among classes. We also imposed that 10% of 
the training set was used for validation purposes.
Please, use the Python notebook labeled "pre-processing" to build the folds.

## Individual Models
Different individual models are built tuning the dropout of the last fully connected layer of the networks. Specifically, for each network, five 
models are built setting the dropout to 0.0 and then ranging it fro, 0.3 to 0.6. Please, use the Python notebook labeled "build_models" to train the models.

## Soft-Voting Ensemble Model
Outcome of the five models is combined using a soft-voting ensemble approach. Please, use the Python notebook labeled "build_models" to assess the ensemble model.


## Hardware and Software requirements
The code has been tested on a IBM AC922 equipped with 512GB and a NVIDIA A100. 
The IBM Watson Machine Learning Community Edition (WML-CE 1.7.0) was used for deployment.

## Reference
[1] He, X.; Wang, S.; Shi, S.; Chu, X.; Tang, J.; Liu, X.; Yan, C.; Zhang, J.; Ding, G. Benchmarking deep learning models and automated
model design for covid-19 detection with chest ct scans. medRxiv 2020.
[2] Zhang, K.; Liu, X.; Shen, J.; Li, Z.; Sang, Y.; et al, X.W. Clinically applicable AI system for accurate diagnosis, quantitative
measurements, and prognosis of COVID-19 pneumonia using computed tomography. Cell 2020, 181, 1423–1433. https://doi.org/10.1016/j.cell.2020.04.045.
