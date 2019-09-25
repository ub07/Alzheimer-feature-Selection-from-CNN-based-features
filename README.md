# Alzheimer Feature Selection from CNN based features using Tunable Gates in Neural Network

Alzheimer Disease is one of the most happening dementia. Mild cognitive impairment is pre stage of Alzheimer. Identifying MCI patients with high risk of AD is crucial factor to reduce disease effect. A CNN approach was taken to create identifiable features from patches of hippocampal region to classify convertible and non-convertible MCI patients. To reduce number of features generated for each image we use Group Feature Selection Multilayer perceptron. Then this features was used in Extreme Learning Machine to classify the subjects.

The dataset was based on ADNI1 Dataset. The ADNI is an ongoing, longitudinal study designed to develop clinical, imaging, genetic, and biochemical biomarkers for the early detection and tracking of AD. The ADNI study began in 2004 and its first 6-year study is called
ADNI1. Dataset include 188 AD, 229 NC, and 401 MCI subjects.


## Steps are sumarised below:

1. Download ADNI1 dataset.
2. Preprocess them using Preprocess.py
3. Remove Brain Degradation due to Age.
4. Create 151 Patches required for CNN.
5. Generate 1024 features from each patch.
6. Use the 1024*151 features in Feature Selection Model.
7. Check the accuracy in Extreme Learning Machine(RBF) Classifier.

For details Please go through the pdf.
