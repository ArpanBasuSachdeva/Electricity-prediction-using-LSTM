import torch
import torch.nn as nn
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from datetime import datetime, timedelta
from meteostat import Point, Hourly
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
# Define the model class
class ElectricityLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(ElectricityLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Reconstruct the model with the same parameters
input_size = 8
hidden_size = 64
num_layers = 2
output_size = 1

model = ElectricityLSTM(input_size, hidden_size, num_layers, output_size)
# Load the state dictionary from the .pkl file
with open(r"C:\Users\Arpan Basu Sachdeva\Downloads\sih\model_state_dict.pkl", 'rb') as f:
    state_dict = pickle.load(f)

# Load the scalers
with open(r"C:\Users\Arpan Basu Sachdeva\Downloads\sih\scaler_X.pkl", 'rb') as f:
    scaler_X = pickle.load(f)

with open(r"C:\Users\Arpan Basu Sachdeva\Downloads\sih\scaler_y.pkl", 'rb') as f:
    scaler_y = pickle.load(f)

# Load the state dictionary into the model
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()
def predict(model, inputs, scaler_X, scaler_y):
    model.eval()  # Set the model to evaluation mode

    # Prepare the input data
    input_data = np.array(inputs).reshape(1, -1)
    input_data = scaler_X.transform(input_data)  # Use transform instead of fit_transform
    input_tensor = torch.FloatTensor(input_data).unsqueeze(0)  # Add batch dimension

    # Make predictions
    with torch.no_grad():
        prediction = model(input_tensor)
    
    # Inverse transform the prediction to get the original scale
    prediction = scaler_y.inverse_transform(prediction.numpy())
    
    return prediction[0][0]


# Set time period
start_date = datetime.now()
end_date = datetime.now() + timedelta(15)

delhi = Point(28.6139, 77.2090)

# Get hourly data
future = Hourly(delhi, start_date, end_date)
future = future.fetch()

# Drop unnecessary columns
future.drop(columns=['snow', 'wpgt', 'tsun'], inplace=True)

# Fill missing values with the mean of each column
future.fillna(future.mean(), inplace=True)

# Print the DataFrame
print(future)

# Traverse each row and print values as a list
for index, row in future.iterrows():
    # Convert the row to a list
    row_values = row.tolist()
    # Print the row values
    print(f"Row index: {index}")
    prediction = predict(model, row_values, scaler_X, scaler_y)
    print(prediction)