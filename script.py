import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class Wattstor_assignment:
    def __init__(self, file_path):
        df = pd.read_csv(file_path, delimiter=';')
        self.data = df

    def get_quantity_name(self, quantity_name):
        return quantity_name


def main():
    parser = argparse.ArgumentParser(description='Regression model')
    parser.add_argument('--input', type=str, help='Input CSV file path')
    parser.add_argument('--quantity', type=str, help='Quantity name for analysis')
    args = parser.parse_args()

    wattstor_assignment = Wattstor_assignment(args.input)
    quantity_name = wattstor_assignment.get_quantity_name(args.quantity) # quantity name

    df = wattstor_assignment.data

    df_copy = df.copy()

    df_copy = df_copy.dropna(axis=0, subset=["Consumption"]) # deletion of NaN value instances
    
    df_copy['Time'] = pd.to_datetime(df_copy['Time'], utc=True) # converting Time string column into pandas datetime object
    df_copy['Year'] = df_copy['Time'].dt.year # separating datetime object into individual columns containing integers
    df_copy['Month'] = df_copy['Time'].dt.month
    df_copy['Day'] = df_copy['Time'].dt.day
    df_copy['Hour'] = df_copy['Time'].dt.hour + df_copy['Time'].dt.minute / 60
    df_copy = df_copy.drop("Time", axis=1) # dropping the Time string column

    X = df_copy.drop(quantity_name, axis=1) 
    y = df_copy[quantity_name]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 

    scaler = StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train) # scaling traning data
    scaled_X_test = scaler.transform(X_test) # scaling testing data

    model = RandomForestRegressor(random_state=42)

    model.fit(scaled_X_train, y_train.ravel()) # fitting to training data

    y_pred = model.predict(scaled_X_test) # prediction of 

    rmse = round(mean_squared_error(y_test, y_pred, squared=False), 2) # calculation of valuation metrics
    mae = round(mean_absolute_error(y_test, y_pred), 2)

    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')

    X_index = df_copy.index.values.reshape(-1, 1)  # indexes to plot measured vs. estimated data
    y_index = df_copy[quantity_name].values.reshape(-1, 1)

    X_train_x, X_test_index, y_train_x, y_test_index = train_test_split(X_index, y, test_size=0.3, random_state=42) # split to match the indexes of estimated data

    plt.figure(figsize=(10, 6))
    plt.scatter(X_index, y_index, color='blue', s = 10, label='Measured Data') # plottin measured data
    plt.scatter(X_test_index, y_pred, color='red', s = 10, label='Fitted Model') # plotting estimated values
    plt.xlabel('Individual instances')
    plt.ylabel(quantity_name)
    plt.title(f'{quantity_name} Prediction')
    plt.xlim(0, 400) # zoom on the first 400 instances for clarity
    plt.legend()
    plt.show()

    #print(idx)



if __name__ == "__main__":
    main()
