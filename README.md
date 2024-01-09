# Combining Convolution and Involution for the Early Prediction of Chronic Kidney Disease

This repository contains code and data samples for the prediction of Chronic Kidney Disease using laboratory analyses.

The `iccs_models.ipynb` file is a jupyter notebook showing an example used for the proposed deep learning models defined
in `models.py`. Three models are defined: one that only uses convolution layers, one that only uses involution layers,
and one that uses a combination of the two.

The `utils.py` file contains several utility functions to load, process and evaluate the data.

The `bio.yml` file contains the dependencies necessary to run the notebook. They can be installed
using [anaconda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment).

Data samples are provided in `data/learning_data.pkl`. The data is already split into training and test samples: the
training set contains 14367 matrices, and the test set contains 1000 analyses.

Each matrix represents a patient's biological history. They are of size `(12,27)`, where 12 is the number of records (
one per month), and 27 the number of analyses.

