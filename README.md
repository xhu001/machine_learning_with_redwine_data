# machine_learning_with_redwine_data
machine learning with redwine data: predicting wine quality ratings based on physicochemical properties

The P. Cortez red wine quality dataset contains data samples of 1599 red wines consisting of 12 physiochemical properties 
(such as alcohol, ph, acidity) and its quality rated by wine connoisseurs (0 – very bad to 10 – very excellent). In this
dataset, quality ranges from 3 to 8.  Details of this dataset can be found at the RED WINE EDA project.

The goal of this project is to create a model using multiclass classification algorithms to predict the quality of a red wine
based on its physiochemical properties.
 
A random 20% split between the training and the validation data (320) is used. Features scaling, Principal component analysis and
Algorithms including decision tree, k-NN, SVM, Naive-Bayes, Random-Forest and Adaboost are tested with parameters tuning.  

The best prediction is gaven by Random-Forest method with Z-score scaling. 
Precision is 0.70; recall is 0.72, f1-score is 0.70.
The confusion matrix is:

          3   4   5   6   7   8
     3 [  0   0   1   0   0   0]
     4 [  0   0   4   3   0   0]
     5 [  0   1 111  16   2   0]
     6 [  0   0  35  94   7   0]
     7 [  0   0   4  15  25   0]
     8 [  0   0   0   2   0   0]


