import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import cross_val_score 
import matplotlib.pyplot as plt 


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
# model and hyperparameter tuning 
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    random_state=42)

# model = XGBRegressor(random_state=42) # compatiblity error occured on this line 

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)
model.fit(X_train , y_train)
prediction = model.predict(X_test)

mae = mean_absolute_error(y_test , prediction)
print(mae)

# Cross-validation scores 
scores = cross_val_score(model , X , y , scoring="neg_mean_absolute_error", cv=5)

mae_scores = -scores
print(mae_scores.mean())


# which features influence house prices the most 

importance = model.feature_importances_
feature_importance = pd.DataFrame({"Feature": X.columns, "Importance": importance})

feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

print(feature_importance.head(10))

# visualising the top features 
top_features = feature_importance.head(10)

plt.figure(figsize=(10, 6))
plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Top 10 Important Features")

plt.gca().invert_yaxis() # Invert y-axis to have the most important feature at the top
plt.savefig("feature_importance.png") # Save the plot as an image file 
plt.show()


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


