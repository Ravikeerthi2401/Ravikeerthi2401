

import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold

cleaned_data = pd.read_csv("C:\\Users\\Ravi Keerthi\\Desktop\\Second sem\\Project-Code\\cleanedAirbnbData.csv")

cleaned_data.drop(['id','Unnamed: 0.1', 'Unnamed: 0', 'review_scores_scaled_rating'], axis=1, inplace=True)

print(cleaned_data.shape)

corr = cleaned_data.corr()

sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns)
plt.show()

# Create correlation matrix
corr_matrix = cleaned_data.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] >= 0.95)]

# drop correlated features
cleaned_data.drop(to_drop, inplace=True, axis=1)

# remove feature with low variance
cleaned_data.drop('requires_license', inplace=True, axis=1)

print(cleaned_data.shape)

data = cleaned_data

data['Positive_Reviews'].hist()
plt.title('Frequency plot for positive reviews')
plt.ylabel("Frequency")
plt.xlabel('No. of Positive Reviews')
plt.show()

data['Negative_Reviews'].hist()
plt.title('Frequency plot for negative reviews')
plt.ylabel("Frequency")
plt.xlabel('No. of Negative Reviews')
plt.show()

data['Neutral_Reviews'].hist()
plt.title('Frequency plot for neutral reviews')
plt.ylabel("Frequency")
plt.xlabel('No. of Neutral Reviews')
plt.show()

y = data['review_scores_rating']

data = data.ix[:, data.columns != 'review_scores_rating']

categorical_cols = ['house_rules', 'host_verifications', 'property_type', 'room_type', 'bed_type',	'amenities',
'cancellation_policy']

# model = lgb.LGBMRegressor(max_depth=5, num_leaves=20, nthread=-1, n_estimators=200)
# score = cross_val_score(model, data, y, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
# print("Scores for LightGBM: ", score)
# print("Mean score for LightGBM: ", score.mean())

x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=2018)


params = {
          'objective': 'regression',
          'nthread': -1,
          'metric' : 'rmse',
         }


# For evaluation of the algorithm
dtrain = lgb.Dataset(x_train, y_train)
model = lgb.train(train_set=dtrain, categorical_feature=categorical_cols, params=params)

y_pred = model.predict(x_test)

print("Mean Absolute Error for the LightGBM Model: ", mean_absolute_error(y_test, y_pred))
print("Root Mean Squared Error for the LightGBM Model", np.sqrt(mean_squared_error(y_test, y_pred)))

x_test['prediction'] = y_pred
x_test.to_csv('save_method_3.csv', index=False)

print('Plot feature importances...')
ax = lgb.plot_importance(model, max_num_features=20)
plt.show()