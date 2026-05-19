import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error 

df = pd.read_csv("data/ames.csv")

df = pd.get_dummies(df, columns=["Neighborhood"], drop_first=True)

df["TotalSF"] = df["TotalBsmtSF"] + df["GrLivArea"]

Neighborhood_cols = [col for col in df.columns if "Neighborhood_" in col]

# modeling section and training data for sale price prediction based on the result of analysis up to this point 

X = df[["TotalSF" ,  "GarageArea", "YearBuilt" , "GarageCars" ,   "LotArea" , "FullBath" , 'LotFrontage' , "TotRmsAbvGrd" , "YearRemodAdd" , "MasVnrArea" , "Fireplaces" 
    , "BsmtFinSF1" , "WoodDeckSF" , "OpenPorchSF"] + Neighborhood_cols]
y = df["SalePrice"]


model = LinearRegression()
model.fit(X, y)

prediction = model.predict(X)

mae = mean_absolute_error(y, prediction)

print(mae)

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


