#WATTSTOR ASSIGNMENT - REGRESSION MODEL

Prerequisities: Python 3.6 or higher
Required Python libraries: pandas, scikit-learn, matplotlib, seaborn

Purpose of this script written in python is to perform regression on a given csv file in a specific format as provided by Wattstor. In this particular case a RandomForestRegressor machine learning model was used.

#USAGE

To run the script, type the following command into terminal

$ python3 script.py --input SG.csv --quantity Consumption

SG.csv - CSV file as an input with the file format same as the sample file (same columns)
Consumption - required quantity name as an input

#OUTPUT

Mean Absolute Error (MAE) of the estimated values
Root Mean Squared Error (RMSE) of the estimated values
Plot displaying the series of measured data together with the values estimated by the ML model.
    x axis - individual instances = measurements every 30 minutes
    y axis - estimated values of chosen quantity

#ADDITIONAL INFORMATION

- rows containing NaN values are being dropped
- original time string column is being replaced by a number of integers describing the time value
- use of traintestsplit
- use of standard scaled
