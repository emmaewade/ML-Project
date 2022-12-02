# Hybrid CNN-XGBoost Model for Video Popularity Prediction
CSE 4683 : Machine Learning and Soft Computing, Fall 2022
### Emma Wade, Michelle Hardin, John Austin Reed

Data and Environment : 
1. *environment.yml* : conda environment, to create environment based on yaml https://edcarp.github.io/introduction-to-conda-for-data-scientists/04-sharing-environments/index.html
2. Data available here: https://bitgrit.net/competition/11 and in project Canvas submission 

Source Code : 
1. *file-prep.py* : prepares training and testing files including one-hot encoding of categorical variables, cycling encoding of time variables, lasso regression of image features, and joining all variables. output needed to run *cnn-xgboost.py* and *XGBOOST.ipynb*
2. *cnn-xgboost.py* : hybrid model and CNN model
3. *XGBOOST.ipynb* : XGBoost model
4. *ML_Plotting.ipynb* : figure and comparisons scripts

