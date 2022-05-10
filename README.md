# Technical Case Study - ML Ops
### Carlo Alberto Barranco 2022-05-08

In this folder I defined a model (using XGBoost) to identify frauds in a dataset from https://www.kaggle.com/kartik2112/fraud-detection-on-paysim-dataset/data then I defined some functions to 
* measure the performance of this model
* call the model so that it can be used on demand to infestigate specific transactions
* ideally automate a pipeline to re-train the model when new verified data are available.

Before running the notebooks the user needs to download the data from the link indicated above and save them in folder "data" as "initial_dataset.csv".

## Model definition
In notebook TechnicalCaseStudy_DefineModel.ipynb there is 
* a study on the original dataset
* choice and redefinition of features useful for prediction
* the definition and training of the model

## Measuring performance
In module functions.model_testing are included several functions used by notebook "TechnicalCaseStudy_TestingModel.ipynb" to measure performance, in particular function "evaluate_prediction" receives the list of actual frauds identifications and the list of predicted frauds (for the same set of transactions) and returns
* True Negative
* False Positive
* False Negative
* True Positive
* accuracy
* precision
* recall

Function "rebalanced_sample_test" receives a dataset, a length and a desired ratio of positive and returns a sub-dataset of the indicated length where positives are in the indicated ratio.

Function "sliding_window_test" runs "evaluate_prediction" on subsets of a given dataset build as sliding windows of given length separated by constant steps of given length. The performance results for each of the windows is saved in a separate row in a given txt file.

## Predictions on demand
In module functions.update_model is included function "call_model_and_predict" which receives in input a pandas data-frame or pandas series indetifying one or more transactions and returns the predictions obtained by the most recent available version of the fraud identification model.

## Re-training pipeline
In folder "dag" is defined dag "update_model_dag.py" which would hypothetically run on Airflow in a Docker and create newer versions of the fraud identification model.

The dag
* verify if there are new data available more recent than last version of the model
* if yes, checks how well current version of the model on those new data
* if performance is above a given threshold, confirms current version
* otherwise it collects more recent data and a given amount of historical data and re-train a new XGBoost model
* stores the new model as a pickle.

The dag relies on new_data being loaded in coherent formats in a given folder.

I don't have Airflow nor Docker on my machine, so I the data pipeline was not tested.

