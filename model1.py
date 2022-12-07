from sklearn import datasets
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

import os

path = os.getcwd()

# Load the Boston Data Set
crime = pd.read_csv("dc-crimes-search-results.csv")

crime['OFFENSE'].value_counts()

crime['offense-text'].value_counts()

crime['offense-text'] = crime['offense-text'].replace("homicide", 0)
crime['offense-text'] = crime['offense-text'].replace("theft f/auto", 1)
crime['offense-text'] = crime['offense-text'].replace("motor vehicle theft", 2)
crime['offense-text'] = crime['offense-text'].replace("theft/other", 3)
crime['offense-text'] = crime['offense-text'].replace("assault w/dangerous weapon", 4)
crime['offense-text'] = crime['offense-text'].replace("robbery", 5)
crime['offense-text'] = crime['offense-text'].replace("burglary", 6)
crime['offense-text'] = crime['offense-text'].replace("sex abuse", 7)
crime['offense-text'] = crime['offense-text'].replace("arson", 8)

crime['SHIFT'] = crime['SHIFT'].replace("midnight", 0)
crime['SHIFT'] = crime['SHIFT'].replace("evening", 1)
crime['SHIFT'] = crime['SHIFT'].replace("day", 2)
crime['METHOD'] = crime['METHOD'].replace("gun", 0)
crime['METHOD'] = crime['METHOD'].replace("others", 1)
crime['METHOD'] = crime['METHOD'].replace("knife", 2)

crime = crime.drop(columns = ['METHOD','XBLOCK','ucr-rank','PSA','WARD','YBLOCK','CENSUS_TRACT','location','VOTING_PRECINCT','BLOCK_GROUP','sector','BID','offensekey','NEIGHBORHOOD_CLUSTER','offensegroup','END_DATE','BLOCK','START_DATE', 'CCN', 'OFFENSE', 'OCTO_RECORD_ID', 'ANC', 'REPORT_DAT'])
crime = crime.dropna()

y = crime['offense-text']

crime = crime.drop(columns = ['offense-text'])

X = crime

print(X)

# Create training and test split
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# Create an instance of Lasso Regression implementation
#
lasso = Lasso(alpha=1.0)
#
# Fit the Lasso model
#
lasso.fit(X_train, y_train)
#

# Create the model score
#
lasso.score(X_test, y_test), lasso.score(X_train, y_train)

row = [-76.930977, 0, 6, 2020, 38.879537]
# make a prediction
yhat = lasso.predict([row])
# summarize prediction

row = [-77.013442, 1, 1, 2020, 38.901699]
# make a prediction
yhat = lasso.predict([row])
# summarize prediction

pickle.dump(lasso, open('model1.pkl','wb'))

model = pickle.load(open('model1.pkl','rb'))
print(model.predict([[-76.967568, 1, 7, 2020, 38.855717]]))