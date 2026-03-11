from data_preprocessing import load_and_split_data
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import pickle

# Get project root folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

processed_path = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(processed_path, exist_ok=True)

x_train, x_test, y_train, y_test = load_and_split_data()

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

pd.DataFrame(x_train_scaled).to_csv(os.path.join(processed_path, "x_train.csv"), index=False)
pd.DataFrame(x_test_scaled).to_csv(os.path.join(processed_path, "x_test.csv"), index=False)
pd.DataFrame(y_train).to_csv(os.path.join(processed_path, "y_train.csv"), index=False)
pd.DataFrame(y_test).to_csv(os.path.join(processed_path, "y_test.csv"), index=False)
with open("artifacts/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Scaler saved in artifacts folder")

print("Feature Engineering Completed")
print("Files saved in:", processed_path)