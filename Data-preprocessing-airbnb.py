# -*- coding: utf-8 -*-


import pandas as pd
table=pd.read_csv("C:\\Users\\Ravi Keerthi\\Desktop\\Second sem\\Project-Code\\listings.csv", sep=",", usecols=['id','host_response_rate',
       'host_is_superhost',
       'is_location_exact', 'property_type', 'room_type', 'accommodates',
       'bathrooms', 'bedrooms', 'beds', 'bed_type', 'amenities', 'square_feet',
       'price','monthly_price', 'security_deposit',
       'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights',
        'number_of_reviews',
       'review_scores_rating',
       'review_scores_accuracy', 'review_scores_cleanliness',
       'review_scores_checkin', 'review_scores_communication',
       'review_scores_location', 'review_scores_value', 'requires_license',
      'cancellation_policy',
        'reviews_per_month','amenities','host_verifications','host_about','house_rules','interaction','description','summary'])

c_list=['host_is_superhost','requires_license', 'is_location_exact']

for col in c_list:
    table[col]=table[col].apply(bool)
    table[col]=table[col].apply(int) 

table['host_response_rate']=table['host_response_rate'].fillna("0") 

roomtype=['Private room', 'Entire home/apt', 'Shared room']
for room in roomtype:
    table.loc[table["room_type"]==room,"room_type"]=roomtype.index(room)+1

ptype=list(table["property_type"].unique())
for prop in ptype:
    table.loc[table["property_type"]==prop,"property_type"]=ptype.index(prop)+1  

btype=list(table["bed_type"].unique())
for bed in btype:
    table.loc[table["bed_type"]==bed,"bed_type"]=btype.index(bed)+1

ctype=list(table["cancellation_policy"].unique())
for canc in ctype:
    table.loc[table["cancellation_policy"]==canc,"cancellation_policy"]=ctype.index(canc)+1
    
table['price'] = table['price'].str.replace('$','')
table['cleaning_fee'] = table['cleaning_fee'].str.replace('$','')
table['extra_people'] = table['extra_people'].str.replace('$','')
table['security_deposit'] = table['security_deposit'].str.replace('$','')    
table['monthly_price'] = table['monthly_price'].str.replace('$','')
table['monthly_price'] = table['monthly_price'].str.replace(',','')
table["host_response_rate"] = table["host_response_rate"].str.replace('%','')

table['house_rules']=table['house_rules'].astype(str)

s = set(["loud", "music", "adults", "children","smoking","shoes","drinking","pets","cleaning","visitors"])

table['house_rules'] = table['house_rules'].str.split(r'[^\w]+')\
                   .apply(lambda x: list(s.intersection(x)))


def conv(string):
    string=str(string)
    string = string.split(',')
    return len(string)

table["house_rules"]=table["house_rules"].apply(conv)
table["amenities"]=table["amenities"].apply(conv)
table["host_verifications"]=table["host_verifications"].apply(conv)


def conv1(string):
    string=str(string)
    string = string.split(' ')
    return len(string)

table["summary"]=table["summary"].apply(conv1)
table["host_about"]=table["host_about"].apply(conv1)
table["interaction"]=table["interaction"].apply(conv1)
table["description"]=table["description"].apply(conv1)

#split = [ 0,10, 20,30,40,50,60,70,80,90,100]
#rating = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
#table['review_scores_scaled_rating'] = pd.cut(table['review_scores_rating'], bins=split, labels=rating)


table['host_response_rate']=table['host_response_rate'].astype(int)
split = [ 0,10, 20,30,40,50,60,70,80,90,100]
bins = [1,2,3,4,5,6,7,8,9,10]
table[['host_response_rate']] = pd.cut(table['host_response_rate'], bins=split, labels=bins)

# fill missing values with mean column values
table['host_response_rate'].fillna(table['host_response_rate'].mean(), inplace=True)

table.to_csv("D:\\MscStudy\\ADM\\Airbnb-dataset\\Berlin\\LatestDatasets\\cleanedAirbnbData.csv")


#Splitign the categories into 50-50
#df = cleanedData[(cleanedData['review_scores_scaled_rating'] == 5)]
#df_split = np.array_split(df, 2)




