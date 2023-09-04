# Detecting Machine Failure from IoT Sensors with a SQL Pipeline

## Overview

Replacing big industrial machines can be very expensive. Performing routine maintenance is critical to prevent serious damage to the equipment and extend its useful life. Therefore, developing the ability to accurately predict when a machine needs maintenance is highly valuable! 

This project uses an SQL pipeline to analyze real-world IoT sensor data from industrial machines. The goal is to detect early signs of machine failure so maintenance can be scheduled proactively before a critical breakdown occurs.

The data comes from sensors installed on the equipment to monitor:

- Vibration
- Pressure
- Temperature

By applying statistical analysis and machine learning algorithms, I identified patterns in the sensor measurements that indicate impending mechanical issues.

The pipeline does the following:

- Load and transform the raw sensor data into an SQL database
- Perform feature engineering to extract relevant metrics from the sensor data  
- Train classification models to detect impending failures based on the sensor metrics
- Evaluate model performance to optimize predictive accuracy
- Deploy the best model to make real-time predictions as new sensor data arrives

This end-to-end SQL-based analytics project provides me with hands-on experience building a workflow for predictive maintenance. The skills learned can be applied to many real-world IoT data streams and use cases.

## Data Overview

The sensor data includes:

- Timestamped readings from multiple sensors on each machine
- Machine operating conditions at the time of each reading 
- Maintenance logs indicating past machine repairs and failures

## Data Exploration

The sensor data is explored through visualizations and summary statistics to understand the properties and relationships between variables. This informs data preprocessing and feature engineering.


## Data Preprocessing

The IoT sensor data is prepared by:

- Splitting sensor metrics into feature and target arrays.
- Handling missing values.
- Removing duplicate readings.  
- Detecting and filtering outliers.
- Rebalancing the data classes using imblearn and SMOTE to handle class imbalance. Oversampling minority classes and undersampling majority classes.

This processing pipeline transforms the raw IoT data into engineered features ready for modeling.


## Feature Engineering  

Principal component analysis (PCA) is applied to reduce the dimensionality of the sensor data. This condenses the many correlated sensor metrics into smaller informative principal components.

The transformed PCA features are engineered as inputs for the machine learning models.

## Model Development

Machine learning models trained in the pipeline include:

- Stochastic Gradient Descent (SGD)
- Ridge Regression
- Random Forest (RF)
- Gradient Boosting (GB) 

These classfiers are trained on the PCA features to predict impending failures. The modeling pipeline involves:

- Pass in StandardScaler() before training PCA
- Train PCA model by setting n_components=50
- Apply ML algorithms by setting random_state

Then, a hyperparameter training grid is created for each classifer:

- Pick hyperparameter(s) to tune for each classifier, and apply a set of values to it

The pipeline of model algorithms is then trained:

- Create a GridSearchCV instance, which allows us to search for a hyperparameter CV, train it, and apply Cross Validation
- Pass in the pipeline, specify the grid, set the number of folds for the cross-validation, and set the number of jobs
- Fit the model against training data X_train and y_train


## Model Evaluation

The trained models are evaluated on key metrics including:

- F1 score
- Precision and recall
- Confusion matrix analysis
- Selecting the best model for production deployment

Performance on the test set indicates real-world effectiveness.

## Deployment

The final model is added back into the SQL pipeline and deployed into production. Predictions run on new live sensor data.

Alerts are generated automatically if failure risk passes a set threshold. This enables timely maintenance. 

New data is fed back to re-train and continuously improve the model.

## Usage

The project provides scripts and documentation to:

- Set up the SQL environment
- Load and prepare the sensor data  
- Train, evaluate, and deploy ML models
- Make predictions on new data
- Visualize model results

## Contributing

Contributions to improve the project are welcome! Please create an issue or PR.
