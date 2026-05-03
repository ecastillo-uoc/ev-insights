**Objective:** Chapter #file:04_resultados.tex  of the thesis. 

To develop this chapter we need to execute the different trainings for the different datasets and using the different feature engineering (use_log_transform, use_differencing,use_calendar_features). 

We are still unsure that all the inner algorithms are correct. We whant to automate as much as possible the execution, artifacts recopilation and chapter #file:04_resultados.tex engineering

I was thinking on refactoring the #file:tfm_forecast.py  module that currently executes 1 case to be able to execute the whole set of cases (combination of datasets, models and feature engineering options, where appropriate), producing each case a set of files in a specific directory in the /home/ecastillo/dev/M2_882_TFM/tfm/doc/vf/chapters path, with an automatically created latex file for the case incorporating the case results, pictures, etc, and then configure the #file:04_resultados.tex  as an aggregator of all these parts. 

The initial executions will help to determine which combinations are interesting and wether the results are reasonable or there may be still problems in the training and prediction agorithm. To achieve this goal we need to produce additional artifacts with enough metadata to help diagnose the 'health' of the case (training, prediction, etc).

1.- Create a plan to review the different algorithms from the point of view of the data science engineer and the correctness of the model, windows, etc, taking into account both the look_back and the forecast_horizon

2.- Define the output artifacts for each case, including metadata to help diagnose the health of the case

3.- Refactor the #tfm_forecast.py adding a new method to run a specific case (data, model, features engineering) and produce the corresponding output artifacts

4.- Add a method to run the whole set of cases derived from the combination of (data, model, features engineering)

5.- Improve #04_resultados.tex to aggregate the cases so the thesis builder includes all the generated outputs.

