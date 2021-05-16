

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

# dataset without features derived from text reviews
cleaned_data.drop(['id','Unnamed: 0.1', 'Unnamed: 0', 'Positive_Reviews', 'Negative_Reviews','Neutral_Reviews', 'review_scores_scaled_rating'], axis=1, inplace=True)

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

# Remove features will low variance
threshold = 0.9
vt = VarianceThreshold().fit(cleaned_data)

# Find feature names
var_threshold = cleaned_data.columns[vt.variances_ < threshold * (1-threshold)]

cleaned_data.drop(var_threshold, inplace=True, axis=1)

print(cleaned_data.shape)

data = cleaned_data

y = data['review_scores_rating']

data = data.ix[:, data.columns != 'review_scores_rating']

# model = lgb.LGBMRegressor(max_depth=5, num_leaves=15, n_estimators=200, objective='regression')
# score = cross_val_score(model, data, y, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
# print("RMSE score for LightGBM: ", (np.sqrt(-score)).mean())
#
# model = RandomForestRegressor(n_estimators=200)
# score = cross_val_score(model, data, y, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
# print("RMSE score for RandomForestRegressor: ", (np.sqrt(-score)).mean())

categorical_cols = ['house_rules', 'host_verifications', 'property_type', 'room_type', 'bed_type',	'amenities',
'cancellation_policy']


a_train, a_test, b_train, b_test = train_test_split(data, y, test_size=0.2, random_state=2018)

params = {
          'objective': 'regression',
          'nthread': -1,
          'metric' : 'rmse',
         }


# For evaluation of the algorithm
dtrain = lgb.Dataset(a_train, b_train)
model = lgb.train(train_set=dtrain, categorical_feature=categorical_cols, params=params)

b_pred = model.predict(a_test)

print("Mean Absolute Error for the LightGBM Model: ", mean_absolute_error(b_test, b_pred))
print("Root Mean Squared Error for the LightGBM Model", np.sqrt(mean_squared_error(b_test, b_pred)))

a_test['prediction'] = b_pred
a_test.to_csv('save_method_2.csv', index=False)

print('Plot feature importances...')
ax = lgb.plot_importance(model, max_num_features=20)
plt.show()