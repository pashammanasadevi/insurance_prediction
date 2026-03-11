# 1. load processed data from processed folder
# 2. create model and train data
# 3. save model in artifacts folder
import os
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression

# Get project root folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, "data", "processed")
artifact_path = os.path.join(BASE_DIR, "artifacts")

# create artifacts folder
os.makedirs(artifact_path, exist_ok=True)

X_train = pd.read_csv(os.path.join(data_path, "x_train.csv"))
y_train = pd.read_csv(os.path.join(data_path, "y_train.csv"))

model = LinearRegression()
model.fit(X_train, y_train)

with open(os.path.join(artifact_path, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully")
print("Model saved in:", artifact_path)