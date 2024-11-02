# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 22:45:52 2024

@author: gadiy
"""

# Importing required libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import statistics as stat
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns







# Importing data
ovarian_data = pd.read_csv("C:/Users/gadiy/Downloads/ovarian_processed.csv")
ovarian_data = pd.DataFrame(ovarian_data)
## Checking column names to verify data import
ovarian_data.columns






# Cleaning data
## Filtering to primary tumor cells
ovarian_processed = ovarian_data[(ovarian_data['primary'] == 1) & (ovarian_data['tissue_category'].str.contains('Tumor'))]

## Turning phenotype str value into a binary value; creating column for each phenotype
ovarian_processed['cd19_bin'] = np.where(ovarian_processed['phenotype_cd19'] == 'CD19+', 1, 0)
ovarian_processed['cd3_bin'] = np.where((ovarian_processed['phenotype_cd3'] == 'CD3+') & (ovarian_processed['phenotype_cd8'] == 'CD8-'), 1, 0)
ovarian_processed['cd8_bin'] = np.where(ovarian_processed['phenotype_cd8'] == 'CD8+', 1, 0)
ovarian_processed['cd68_bin'] = np.where(ovarian_processed['phenotype_cd68'] == 'CD68+', 1, 0)

## Checking dimensions of filtered objects to verify filtering data cleaning
ovarian_processed.shape





# Calculating nearest neighbor statistics for cd8-cd68 spatial correlation
## Bivariate g function, taken from Vectra Polaris tibble 
def g_function(distances, max_distance):
    g = np.zeros(len(distances))
    for i, d in enumerate(distances):
        g[i] = np.sum(d <= max_distance)
    return g

## Creating cell-specific data frames
cd8 = ovarian_processed[ovarian_processed['cd8_bin'] == 1]
cd68 = ovarian_processed[ovarian_processed['cd68_bin'] == 1]

## Looping through to get the nearest neighbor G value for each sample using their cd8/cd68 populations
ovarian_processed['g_value'] = 0
for val in ovarian_processed['sample_id_int'].unique():
    # Filtering to cells within the sample of interest 
    cd8_filt = cd8[cd8['sample_id_int'] == val]
    cd68_filt = cd68[cd68['sample_id_int'] == val]
    
    ## Creating a point-process matrix using x-y values for cd8 and cd68
    distances = distance_matrix(cd8_filt[['x', 'y']], cd68_filt[['x', 'y']])
    
    ## Calculating minimum nearest neighbor distance
    nearest_neighbor_distances = np.min(distances, axis = 1)
    
    ## Calculating range of radii of which to calculate nearest neighbors
    max_distances = np.arange(5, 250, 5)
    
    ## Calculating bivariate nearest neighbor G values
    g_values = [g_function(nearest_neighbor_distances, d) for d in max_distances]
    
    ### [0] extracts the first array from the tuple while [0] extracts the index of the desired element
    if len(g_values[1]) >= 1:
        g_100 = stat.mean(g_values[np.where(max_distances == 100)[0][0]])
    else:
        g_100 = 0
    
    ## Creating a column for g values for a sample at different radii
    ovarian_processed['g_value'] = np.where(ovarian_processed['sample_id_int'] == val, g_100, ovarian_processed['g_value'])






# Creating a logistic regression model to predict survival using cd8-cd68 nearest neighbor values and clinical covariates
## Splitting data into features and target variable
feature_cols = ['recurrent', 'age_at_diagnosis', 'stage', 'g_value']
x = ovarian_processed[feature_cols]
y = ovarian_processed.death

## Split X and Y into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.40, random_state=16)

## Insantiate the model
logreg = LogisticRegression(random_state = 16)

## Fitting model with data
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

## Calculating confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)


## Adding aesthetics to confusion matrix plot
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# Creating heatmap for the confusion matrix plot
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix w/ G-value', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')



# Creating a logistic regression model to predict survival without cd8-cd68 nearest neighbor values
## Splitting data into features and target variable
feature_cols = ['recurrent', 'age_at_diagnosis', 'stage']
x2 = ovarian_processed[feature_cols]
y2 = ovarian_processed.death

## Split X and Y into training and test sets
X_train2, X_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=0.40, random_state=16)

## Insantiate the model
logreg2 = LogisticRegression(random_state = 16)

## Fitting model with data
logreg2.fit(X_train2, y_train2)
y_pred2 = logreg2.predict(X_test2)

## Calculating confusion matrix
cnf_matrix2 = metrics.confusion_matrix(y_test2, y_pred2)


## Adding aesthetics to confusion matrix plot
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# Creating heatmap for the confusion matrix plot
sns.heatmap(pd.DataFrame(cnf_matrix2), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix w/o G-value', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')






