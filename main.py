# import 
import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import GridSearchCV 
# from xgboost import XGBRegressor 

#### load the dataset and perform feature engineering #### 
df = pd.read_csv("data/ames.csv")

# =========== Missing Value Handling ===========
# =========== Filling missing numerical values with median ===========
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

#### One-hot encode categorical values #### 
df = pd.get_dummies(df, columns=["Neighborhood"], drop_first=True)

# create total square footage feature 
df["TotalSF"] = df["TotalBsmtSF"] + df["GrLivArea"]

#### store encoded neighborhood columns #### 
Neighborhood_cols = [col for col in df.columns if "Neighborhood_" in col]

#### feature selection selecting features used for training #### 

X = df[["TotalSF" ,  "GarageArea", "YearBuilt" , "GarageCars" ,   "LotArea" , "FullBath" , 'LotFrontage' , "TotRmsAbvGrd" , "YearRemodAdd" , "MasVnrArea" , "Fireplaces" 
    , "BsmtFinSF1" , "WoodDeckSF" , "OpenPorchSF"] + Neighborhood_cols]
y = df["SalePrice"]

#### Machine Learning models ####
# ====== model performace comparison ======

# Linear regression 
lr_model = LinearRegression()

#### Optimised Random Forest ####
rf_model = RandomForestRegressor(
    n_estimators=200,  # number of trees in the forest
    max_depth=20,      # maximum depth of the trees
    min_samples_split=5, # minimum samples required to split an internal node
    random_state=42
)

#### Gradient Boosting Regressor ####
gbr_model = GradientBoostingRegressor(random_state=42)

#### baseline linear regression model #### 

# model = LinearRegression() 

#### random forest model and automated hyperparameter tuning #### 

#### Train/Test split #### 
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)

#### reusable evalution function #### 
def evaluate_model(model, model_name):
    
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    mae = mean_absolute_error(y_test, prediction )

    scores = cross_val_score(model, X, y, scoring="neg_mean_absolute_error", cv=5)

    cv_mae = -scores.mean()

    print(f"{model_name}")
    print(f"Train/Test MAE: {mae:.2f}")
    print(f"Cross-Validation MAE: {cv_mae:.2f}")
    print("---------------------")

#====== Runing All Models for Comparison ======

evaluate_model(lr_model, "Linear Regression")
evaluate_model(rf_model, "Random Forest")
evaluate_model(gbr_model, "Gradient Boosting Regressor")

####=== base model ###=== 

# rf = RandomForestRegressor(random_state=42)

# model = RandomForestRegressor(
#     n_estimators=200,  # number of trees in the forest
#     max_depth=20,      # maximum depth of the trees
#     min_samples_split=5, # minimum samples required to split an internal node
#     random_state=42
# )

#### Hyperparameter grid #### 

# param_grid = {
#     "n_estimators": [100, 200],
#     "max_depth": [10, 20],
#     "min_samples_split": [2, 5]
# }

####==== Grid search ###==== 

# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring="neg_mean_absolute_error",
#                            n_jobs=-1)

# grid_search.fit(X_train, y_train)



#### Best model after hyperparameter tuning #### 

# model = grid_search.best_estimator_

# print("Best Parameters:", grid_search.best_params_)
# print("Best CV Score:", grid_search.best_score_)

#### XGBoost model #### 
# model = XGBRegressor(random_state=42) # currently disabled due to compatibility error




#### Model Training/Fit the Model ###
# model.fit(X_train , y_train)

###prediction and evaluation ### 

# prediction = model.predict(X_test)


###==== residual analysis ###==== 

# residuals = y_test - prediction
# print(residuals.head())

### residual visualization ### 
# plt.figure(figsize=(8, 6))

# plt.scatter(prediction, residuals)

# plt.axhline(y=0, color='red', linestyle='--')

# plt.xlabel("Predicted Prices")
# plt.ylabel("Residuals")

# plt.title("Residual Plot")

# plt.savefig("residual_plot.png") # Save the plot as an image file
# plt.show()

# mae = mean_absolute_error(y_test , prediction)
# print(mae)

#####===== Cross-validation Evaluating model performance across multiple splits ###==== 

# scores = cross_val_score(model , X , y , scoring="neg_mean_absolute_error", cv=5)
# mae_scores = -scores
# print(mae_scores.mean())


####==== Feature Importance Analysis -which features influence house prices the most ###==== 

# importance = model.feature_importances_
# feature_importance = pd.DataFrame({"Feature": X.columns, "Importance": importance})

# feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

# print(feature_importance.head(10))

###==== visualising the top features/based on Features importance ###=====
# top_features = feature_importance.head(10)

# plt.figure(figsize=(10, 6))
# plt.barh(
#     top_features["Feature"],
#    top_features["Importance"]
# )
# plt.xlabel("Importance")
# plt.ylabel("Features")
# plt.title("Top 10 Important Features")
# plt.gca().invert_yaxis() # Invert y-axis to have the most important feature at the top
# plt.savefig("feature_importance.png") # Save the plot as an image file 
# plt.show()


###==== predicted vs actual scatter plot ###=====

# plt.figure(figsize=(8, 8))

# plt.scatter(y_test, prediction)

# plt.xlabel("Actual Sale Price")
# plt.ylabel("Predicted Sale Price")

# plt.title("Actual vs Predicted House Price")
# plt.savefig("actual_vs_predicted.png") # Save the plot as an image file
# plt.show()


###===== Correlation heatmap for exploring relationships between features##====
# correlation = df [[
#     "SalePrice",
#     "TotalSF",
#     "GarageCars",
#     "GarageArea",
#     "YearBuilt",
#     "FullBath",
#     "LotArea"
# ]].corr()

# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation, annot=True, cmap="coolwarm")

# plt.title("Correlation Heatmap")
# plt.savefig("correlation_heatmap.png") # Save the plot as an image file 
# plt.show()

# print(df.isnull().sum()) ==== temporary inspection/debugging step of missing values,
# had to comment out correlation heatmap and feature importance plot and vice versa 




###===== Exploratory Data Analysis (EDA) , useful dataset & inspection commands ###=====

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


