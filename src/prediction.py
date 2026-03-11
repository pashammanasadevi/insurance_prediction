# 1. Load scaler.pkl and model.pkl
# 2. Get the inputs from the user
# 3. Scale the inputs
# 4. Predict the output
# 5. Print the output

import pickle
import numpy as np

class Insurance_Prediction:
    def __init__(self):
        with open("artifacts/scaler.pkl","rb") as f:
            self.scaler=pickle.load(f)
        with open("artifacts/model.pkl","rb") as f:
            self.model=pickle.load(f)

    def prediction(self,Age,Annual_Income_LPA,Policy_Term_Years,Sum_Assured_Lakhs):
        input=np.array([Age,Annual_Income_LPA,Policy_Term_Years,Sum_Assured_Lakhs]).reshape(1,-1)
        input=self.scaler.transform(input)
        result=self.model.predict(input)
        return result[0]