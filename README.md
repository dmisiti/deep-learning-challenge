# Deep Learning - Predicting Successful Funding Applicants

## Analysis Overview
In this project, machine learning was used to support the applicant funding process for Alphabet Soup, a nonprofit foundation that has previously supported over 34,000 organizations. TensorFlow was used to design a deep learning model that could be used for binary classification, ultimately predicting whether or not an applicant would be successful if they received funding.

## Results

### Data Preprocessing
- Target variable: `IS_SUCCESSFUL`
- Feature variables (8): `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, `ASK_AMT`
- Removed variables (2): `EIN`, `NAME`

Overall, the variables representing identification columns can be removed, and the feature variables will help the model determine if an applicant will be successful or not, based in part on past success of previous applicants.

### Compiling, Training, and Evaluating the Model

Initial Model:
- Hidden Layer 1: 80 neurons (relu activation)
- Hidden Layer 2: 30 neurons (relu activation)
- Output Layer (sigmoid activation)
- Epochs: 100
- Accuracy: 0.7314
- Loss: 0.6002

Two hidden layers were used, keeping a balanced between expressiveness and training efficiency. The step-down from 80 to 30 neurons helps in feature selection and avoiding overfitting. ReLU activiation was used for hidden layers to ensure fast and efficient training, and sigmoid for the output layer to produce probabilities.

Optimized Model:
- Hidden Layer 1: 80 neurons (leaky relu activation)
- Hidden Layer 2: 30 neurons (relu activation)
- Hidden Layer 3: 20 neurons (relu activation)
- Output Layer (sigmoid activation)
- Epochs: 150
- Accuracy: 0.7339
- Loss: 0.5963

The extra hidden layer allows the network to capture more abstract patterns, improving expressiveness while remaining computationally reasonable. Using Leaky ReLU in the first layer helps ensure all neurons contribute to learning. Neurons continue to reduce by layer to keep overfitting risk reduced. Epochs were increased by 50% to provide more exposure to the training data.

Additionally, the optimized model slightly altered the binning categories, creating more bins in the Application Type and Classification categories in an effort to more particularly represent different applicant categories.

Overall, after several different iterations of the optimized model that lead to this final product, the optimized model showed only a marginal increase in accuracy, and did not reach the target accuracy of 75%. Many different combinations of nodal and neuronal numbers, data category and feature alterations, activation types, and epoch volume were used to try to reach this mark, with challenges to get above even 74%.

## Summary

An accuracy of 73.39% in the optimzed model suggests that this model can correctly predict a funding applicant to be successful or unsuccessful in approximately 3 out of every 4 instances.

To improve performance, the model's complexity was increased by adding a third hidden layer and switching the activation function in the first hidden layer from ReLU to Leaky ReLU. Additionally, the training duration was extended from 100 to 150 epochs to allow for more weight adjustments. These changes aimed to enhance the model's ability to capture complex patterns.

A potential alternative for this classification task would be to use a Random Forest Classifier instead of a neural network. These models are less sensitve to small variations in data, work well with tabular data, and provides more direct insights into the relative importance of each feature.
