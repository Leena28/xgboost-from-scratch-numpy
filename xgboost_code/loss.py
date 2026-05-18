import numpy as np
import pandas as pd

df=pd.read_csv(r"california_housing.csv")
x=df.drop(columns=["MedHouseVal"]).values
y_true=df["MedHouseVal"].values

y_pred = np.zeros(len(y_true))


def compute_gradients_and_hessians(y_true, y_pred):
     
    gradients = 2 * (y_pred - y_true)
    hessians = np.full_like(y_true, 2)

    return gradients, hessians


gradients, hessians = compute_gradients_and_hessians(y_true, y_pred)

# print(gradients[:5])
# print(hessians[:5])
