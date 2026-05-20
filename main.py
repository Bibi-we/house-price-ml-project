import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
# from xgboost import XGBRegressor 

df = pd.read_csv("data/ames.csv")

df = pd.get_dummies(df, columns=["Neighborhood"], drop_first=True)

df["TotalSF"] = df["TotalBsmtSF"] + df["GrLivArea"]

Neighborhood_cols = [col for col in df.columns if "Neighborhood_" in col]

# modeling section and training data for sale price prediction based on the result of analysis up to this point 

X = df[["TotalSF" ,  "GarageArea", "YearBuilt" , "GarageCars" ,   "LotArea" , "FullBath" , 'LotFrontage' , "TotRmsAbvGrd" , "YearRemodAdd" , "MasVnrArea" , "Fireplaces" 
    , "BsmtFinSF1" , "WoodDeckSF" , "OpenPorchSF"] + Neighborhood_cols]
y = df["SalePrice"]

# ML models 

# model = LinearRegression() 
model = RandomForestRegressor(random_state=42)
# model = XGBRegressor(random_state=42) # compatiblity error occured on this line 

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)
model.fit(X_train , y_train)
prediction = model.predict(X_test)

mae = mean_absolute_error(y_test , prediction)
print(mae)

# which features influence house prices the most 

importance = model.feature_importances_
feature_importance = pd.DataFrame({"Feature": X.columns, "Importance": importance})

feature_importance = feature_importance.sort_values(by="Importance", ascending=False)
print(feature_importance)

# print(df.select_dtypes(include=["object"]).columns)
# print a list numerical features 
# print(df.select_dtypes(include=["int64" , "float64"]).columns)
# print a list of categorical features 
# print(prediction[:5])
# print(df.head())
# print(df.shape)
# print(df.columns)
# print(df.info())
# print(df.describe())
#print(df["SalePrice"].describe())
#print(df.isnull().sum())
# print(df[["TotalBsmtSF" , "GrLivArea" , "TotalSF"]].head())


