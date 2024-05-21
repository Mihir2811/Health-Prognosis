
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
import pickle

"""Data Collection and Processing"""

# loading the csv data to a Pandas DataFrame
heart_data = pd.read_excel('Heart_Attack.xlsx')



# print first 5 rows of the dataset
heart_data.head()

# print last 5 rows of the dataset
heart_data.tail()

# number of rows and columns in the dataset
heart_data.shape

# getting some info about the data
heart_data.info()

# checking for missing values
heart_data.isnull().sum()

# statistical measures about the data
heart_data.describe()

# checking the distribution of Target Variable
heart_data['target'].value_counts()

"""1 --> Defective Heart

0 --> Healthy Heart

Splitting the Features and Target
"""

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

#print(X)

#print(Y)

"""Splitting the Data into Training data & Test Data"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

#print(X.shape, X_train.shape, X_test.shape)

"""Model Training

Logistic Regression
"""

model = LogisticRegression(C=0.1, penalty='l2', solver='liblinear', random_state=42)

# training the LogisticRegression model with Training data
model.fit(X_train.values, Y_train.values)

"""Model Evaluation

Accuracy Score
"""

# accuracy on training data
X_train_prediction = model.predict(X_train.values)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test.values)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Test data : ', test_data_accuracy)

"""Building a Predictive System"""

input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2) #have attack

input_data2 = (63,1,3,150,268,1,1,187,0,3.6,0,2,2) #does not have an attack

data = input_data2
final= [np.array(data)]

prediction = model.predict(final)
print(prediction)

if (prediction[0]== 0):
    print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')

pickle.dump(model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))