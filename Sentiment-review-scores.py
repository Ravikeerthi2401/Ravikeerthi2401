

import pandas as pd
from textblob import TextBlob

data = pd.read_csv(C:\\Users\\Ravi Keerthi\\Desktop\\Second sem\\Project-Code\\reviews.csv", encoding="ISO-8859-1")
cleaned_data = pd.read_csv("D:\\MscStudy\\ADM\\Airbnb-dataset\\Berlin\\LatestDatasets\\cleanedAirbnbData.csv")

cleaned_data.drop('Unnamed: 0', axis=1, inplace=True)

print(data.shape)
data['polarity'] = data['comments'].apply(lambda x: TextBlob(str(x)).polarity)
data['subjectivity'] = data['comments'].apply(lambda x: TextBlob(str(x)).subjectivity)
print(data.head(5))

data_positive = data.groupby(['listing_id']).agg({'polarity': lambda x: sum(x>0)}).add_suffix('_Positive').reset_index()
data_negative = data.groupby(['listing_id']).agg({'polarity': lambda x: sum(x<0)}).add_suffix('_Negative').reset_index()
data_neutral = data.groupby(['listing_id']).agg({'polarity': lambda x: sum(x==0)}).add_suffix('_Neutral').reset_index()

result = pd.merge(data_positive, data_negative, on='listing_id')
Finalresult = pd.merge(result, data_neutral, on='listing_id')

Finalresult.to_csv('opinionAnalysisPython.csv', index=False)
