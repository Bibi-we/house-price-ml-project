# import 
import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import cross_val_score 
import matplotlib.pyplot as plt 
import seaborn as sns 

# from xgboost import XGBRegressor 

# load the dataset and perform feature engineering 
df = pd.read_csv("data/ames.csv")

# One-hot encode categorical values 
df = pd.get_dummies(df, columns=["Neighborhood"], drop_first=True)

# create total square footage feature 
df["TotalSF"] = df["TotalBsmtSF"] + df["GrLivArea"]

# store encoded neighborhood columns 
Neighborhood_cols = [col for col in df.columns if "Neighborhood_" in col]

# feature selection selecting features used for training 

X = df[["TotalSF" ,  "GarageArea", "YearBuilt" , "GarageCars" ,   "LotArea" , "FullBath" , 'LotFrontage' , "TotRmsAbvGrd" , "YearRemodAdd" , "MasVnrArea" , "Fireplaces" 
    , "BsmtFinSF1" , "WoodDeckSF" , "OpenPorchSF"] + Neighborhood_cols]
y = df["SalePrice"]

#### Machine Learning models ####

# baseline linear regression model 
# model = LinearRegression() 

# random forest model and hyperparameter tuning 
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    random_state=42)

# XGBoost model 
# model = XGBRegressor(random_state=42) # currently disabled due to compatibility error

# Train/Test split
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)

# Model Training/Fit the Model
model.fit(X_train , y_train)

# prediction and evaluation 
prediction = model.predict(X_test)
mae = mean_absolute_error(y_test , prediction)
print(mae)

# Cross-validation Evaluating model performance across multiple splits 
scores = cross_val_score(model , X , y , scoring="neg_mean_absolute_error", cv=5)
mae_scores = -scores
print(mae_scores.mean())


# Feature Importance Analysis -which features influence house prices the most 
importance = model.feature_importances_
feature_importance = pd.DataFrame({"Feature": X.columns, "Importance": importance})

feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

print(feature_importance.head(10))

# visualising the top features/based on Features importance 
top_features = feature_importance.head(10)

# plt.figure(figsize=(10, 6))
# plt.barh(
    #top_features["Feature"],
   # top_features["Importance"]
# )
# plt.xlabel("Importance")
# plt.ylabel("Features")
# plt.title("Top 10 Important Features")
# plt.gca().invert_yaxis() # Invert y-axis to have the most important feature at the top
# plt.savefig("feature_importance.png") # Save the plot as an image file 
# plt.show()


### predicted vs actual scatter plot ###

# plt.figure(figsize=(8, 8))

# plt.scatter(y_test, prediction)

# plt.xlabel("Actual Sale Price")
# plt.ylabel("Predicted Sale Price")

# plt.title("Actual vs Predicted House Price")
# plt.savefig("actual_vs_predicted.png") # Save the plot as an image file
# plt.show()


# Correlation heatmap for exploring relationships between features
correlation = df [[
    "SalePrice",
    "TotalSF",
    "GarageCars",
    "GarageArea",
    "YearBuilt",
    "FullBath",
    "LotArea"
]].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap="coolwarm")

plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png") # Save the plot as an image file 
plt.show()



### Exploratory Data Analysis (EDA) , useful dataset & inspection commands ###

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
# print(df["SalePrice"].describe())
# print(df.isnull().sum())
# print(df[["TotalBsmtSF" , "GrLivArea" , "TotalSF"]].head())


