from tensorflow.keras.models import load_model
import pickle
import pandas as pd


model = load_model("./model/model.keras")

with open('./model/ocp_onehot_encoder.pkl', 'rb') as f1:
    ocp_onehot_encoder = pickle.load(f1)

with open('./model/X_standard_scaler.pkl', 'rb') as f2:
   X_standard_scaler = pickle.load(f2)
   
with open('./model/X_robust_scaler.pkl', 'rb') as f3:
   X_robust_scaler = pickle.load(f3)
   
with open('./model/Y_standard_scaler.pkl', 'rb') as f4:
   Y_standard_scaler = pickle.load(f4)
   
with open('./model/data_preprocessor.pkl', 'rb') as f5:
   data_preprocessor = pickle.load(f5)


def prepare_input(input): 
    X_col_names = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity']
       
    if not isinstance(input, pd.DataFrame):
        X_predict = pd.DataFrame(
            input,
            columns = X_col_names,
        )
        
    X_predict["ocean_proximity"] = X_predict["ocean_proximity"].str.strip().str.lower()

    X_ocp = ocp_onehot_encoder.transform(X_predict[["ocean_proximity"]]) # (-1, 5)
    
    X_ocp = pd.DataFrame(
        X_ocp,
        columns = ocp_onehot_encoder.get_feature_names_out(
            ['ocean_proximity']
        ),
        index = X_predict.index
    )
    
    normal_col_names = [
        "longitude",
        "latitude",
        "housing_median_age",
        "median_income"
    ]
    
    skew_col_names = [
        "total_rooms",
        "total_bedrooms",
        "population",
        "households"
    ]
    
    X_cols_rest = pd.DataFrame(
        data_preprocessor.transform(X_predict.drop(columns=["ocean_proximity"])),  # transform
        columns=X_predict.drop(columns=["ocean_proximity"]).columns,
        index=X_predict.index
    )
     
    X_predict = pd.concat([X_cols_rest, X_ocp], axis=1) # (-1, 13)
       
    return X_predict

def predict(X_predict):
    Y_predict = model.predict(X_predict)
    return Y_standard_scaler.inverse_transform(Y_predict)

input = [
    {
        "longitude": -121.22,
        "latitude": 39.43,
        "housing_median_age": 17.0,
        "total_rooms": 2254.0,
        "total_bedrooms": 485.0,
        "population": 1007.0,
        "households": 433.0,
        "median_income": 1.7000,
        "ocean_proximity": "INLAND"
        # "median_house_value": 92300.0
    }, 
    {
        "longitude": -122.25,
        "latitude": 37.85,
        "housing_median_age": 52.0,
        "total_rooms": 1627.0,
        "total_bedrooms": 280.0,
        "population": 565.0,
        "households": 259.0,
        "median_income": 3.8462,
        "ocean_proximity": "NEAR BAY",
        # "median_house_value": 342200.0
    },
    {
        "longitude": -122.2300,
        "latitude": 37.8800,
        "housing_median_age": 41.0000,
        "total_rooms": 880.0000,
        "total_bedrooms": 129.0000,
        "population": 322.0000,
        "households": 126.0000,
        "median_income": 8.3252,
        "ocean_proximity": "ISLAND",
        # "median_house_value": 452600.0000
    }
]

X_predict = prepare_input(input)

Y_predict = predict(X_predict)

print(f"Predicted median_house_value: {Y_predict}")
