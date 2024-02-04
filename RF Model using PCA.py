import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

Dataset = pd.read_excel('Data.xlsx')

y = Dataset[['Glucose con', '5-HMF yield', 'Select']]
x = Dataset.drop(['Catalysts', 'Glucose con', '5-HMF yield', 'Select'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

y_train_1, y_test_1 = y_train['Glucose con'], y_test['Glucose con']
y_train_2, y_test_2 = y_train['5-HMF yield'], y_test['5-HMF yield']
y_train_3, y_test_3 = y_train['Select'], y_test['Select']
n = 4
pipeline_1 = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=n)),
    ('rf', RandomForestRegressor(n_estimators=20, random_state=0, max_depth=2))
])

pipeline_1.fit(x_train, y_train_1)

y_pred_1 = pipeline_1.predict(x_test)
mse_1 = mean_squared_error(y_test_1, y_pred_1)
r2_1 = r2_score(y_test_1, y_pred_1)
rmse_1 = np.sqrt(mse_1)

print(f'Predict Glucose Conversion')
print(f'Mean Squared Error: {mse_1}')
print(f'R-squared: {r2_1}')
print(f'Root Mean Squared Error: {rmse_1}')

pipeline_2 = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=n)),
    ('rf', RandomForestRegressor(n_estimators=20, random_state=0, max_depth=2))
])

pipeline_2.fit(x_train, y_train_2)

y_pred_2 = pipeline_2.predict(x_test)
mse_2 = mean_squared_error(y_test_2, y_pred_2)
r2_2 = r2_score(y_test_2, y_pred_2)
rmse_2 = np.sqrt(mse_2)

print(f'Predict HMF Yield')
print(f'Mean Squared Error: {mse_2}')
print(f'R-squared: {r2_2}')
print(f'Root Mean Squared Error: {rmse_2}')

pipeline_3 = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=n)),
    ('rf', RandomForestRegressor(n_estimators=20, random_state=0, max_depth=2))
])

pipeline_3.fit(x_train, y_train_3)

y_pred_3 = pipeline_2.predict(x_test)
mse_3 = mean_squared_error(y_test_3, y_pred_3)
r2_3 = r2_score(y_test_3, y_pred_3)
rmse_3 = np.sqrt(mse_3)

print(f'Predict HMF Selec')
print(f'Mean Squared Error: {mse_3}')
print(f'R-squared: {r2_3}')
print(f'Root Mean Squared Error: {rmse_3}')

# Set a fixed square figure size
plt.figure(figsize=(20, 10))

# Plotting for Glucose Conversion - Train Data
plt.subplot(2, 3, 1)
y_train_pred_1 = pipeline_1.predict(x_train)
plt.scatter(y_train_1, y_train_pred_1)
plt.plot([min(y_train_1), max(y_train_1)], [min(y_train_1), max(y_train_1)], '--', color='red')  # y=x line
plt.title('Glucose Conversion: Train Data')
plt.xlabel('Actual Glucose Conversion (Train)')
plt.ylabel('Predicted Glucose Conversion (Train)')
plt.grid(True)
plt.text(0.5, 0.1, f'R2 = {r2_score(y_train_1, y_train_pred_1):.3f}\nRMSE = {np.sqrt(mean_squared_error(y_train_1, y_train_pred_1)):.3f}', transform=plt.gca().transAxes, color='red')

# Plotting for 5-HMF Yield - Train Data
plt.subplot(2, 3, 2)
y_train_pred_2 = pipeline_2.predict(x_train)
plt.scatter(y_train_2, y_train_pred_2)
plt.plot([min(y_train_2), max(y_train_2)], [min(y_train_2), max(y_train_2)], '--', color='red')  # y=x line
plt.title('5-HMF Yield: Train Data')
plt.xlabel('Actual 5-HMF Yield (Train)')
plt.ylabel('Predicted 5-HMF Yield (Train)')
plt.grid(True)
plt.text(0.5, 0.1, f'R2 = {r2_score(y_train_2, y_train_pred_2):.3f}\nRMSE = {np.sqrt(mean_squared_error(y_train_2, y_train_pred_2)):.3f}', transform=plt.gca().transAxes, color='red')

# Plotting for HMF Select - Train Data
plt.subplot(2, 3, 3)
y_train_pred_3 = pipeline_3.predict(x_train)
plt.scatter(y_train_3, y_train_pred_3)
plt.plot([min(y_train_3), max(y_train_3)], [min(y_train_3), max(y_train_3)], '--', color='red')  # y=x line
plt.title('HMF Select: Train Data')
plt.xlabel('Actual HMF Select (Train)')
plt.ylabel('Predicted HMF Select (Train)')
plt.grid(True)
plt.text(0.5, 0.1, f'R2 = {r2_score(y_train_3, y_train_pred_3):.3f}\nRMSE = {np.sqrt(mean_squared_error(y_train_3, y_train_pred_3)):.3f}', transform=plt.gca().transAxes, color='red')

# Plotting for Glucose Conversion - Test Data
plt.subplot(2, 3, 4)
plt.scatter(y_test_1, y_pred_1)
plt.plot([min(y_test_1), max(y_test_1)], [min(y_test_1), max(y_test_1)], '--', color='red')  # y=x line
plt.title('Glucose Conversion: Test Data')
plt.xlabel('Actual Glucose Conversion')
plt.ylabel('Predicted Glucose Conversion')
plt.grid(True)
plt.text(0.5, 0.1, f'R2 = {r2_1:.3f}\nRMSE = {rmse_1:.3f}', transform=plt.gca().transAxes, color='red')

# Plotting for 5-HMF Yield - Test Data
plt.subplot(2, 3, 5)
plt.scatter(y_test_2, y_pred_2)
plt.plot([min(y_test_2), max(y_test_2)], [min(y_test_2), max(y_test_2)], '--', color='red')  # y=x line
plt.title('5-HMF Yield: Test Data')
plt.xlabel('Actual 5-HMF Yield')
plt.ylabel('Predicted 5-HMF Yield')
plt.grid(True)
plt.text(0.5, 0.1, f'R2 = {r2_2:.3f}\nRMSE = {rmse_2:.3f}', transform=plt.gca().transAxes, color='red')

# Plotting for HMF Select - Test Data
plt.subplot(2, 3, 6)
plt.scatter(y_test_3, y_pred_3)
plt.plot([min(y_test_3), max(y_test_3)], [min(y_test_3), max(y_test_3)], '--', color='red')  # y=x line
plt.title('HMF Select: Test Data')
plt.xlabel('Actual HMF Select')
plt.ylabel('Predicted HMF Select')
plt.grid(True)
plt.text(0.5, 0.1, f'R2 = {r2_3:.3f}\nRMSE = {rmse_3:.3f}', transform=plt.gca().transAxes, color='red')

plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.suptitle('RF with PCA')
plt.savefig('RF_with_PCA_Results.png')
plt.tight_layout()