# PR-P
Ride-hailing Service Pattern Recognition and Demand Prediction: A Reinforcement Ensemble Learning with Fuzzy C-Means Clustering Approach

This project analyzes ride-hailing demand data collected from John F. Kennedy International Airport (JFK), New York, spanning from January 1, 2024, to August 31, 2024.

Data Collection: The dataset contains ride-hailing demand data from JFK Airport over the specified period.

Noise Reduction: singular spectrum analysis (SSA) was applied to denoise the data, retaining 90% of the signal’s branches.

Phase Space Reconstruction: The optimal time-delay for phase space reconstruction was determined to be 7 based on the analysis.

 

Clustering: Weighted fuzzy c-means (WFCM) clustering was performed by maximizing both the silhouette coefficient and the fuzzy partition coefficient, identifying two distinct clusters with clear patterns. New hourly data points were classified based on their membership values.

 

Prediction: 
	A single model was first trained, with 90% of the data used for training and validation the remaining 10% for test, using cross-validation.
	Reinforcement learning was then applied to select the optimal predictor.
	The final ensemble model was built using a multi-layer perceptron (MLP), with hyperparameters optimized via Bayesian optimization.
