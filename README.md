# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural network regression is a supervised learning method, and therefore requires a tagged dataset, which includes a label column. Because a regression model predicts a numerical value, the label column must be a numerical data type. You can train the model by providing the model and the tagged dataset as an input to Train Model.

In this experiment we need to develop a Neural Network Regression Model so first we need to create a dataset (i.e : an excel file with some inputs as well as corresponding outputs).Then upload the sheet to drive then using corresponding code open the sheet and then import the required python libraries for porcessing.

Use df.head to get the first 5 values from the dataset or sheet.Then assign x and y values for input and coressponding outputs.Then split the dataset into testing and training,fit the training set and for the model use the "relu" activation function for the hidden layers of the neural network (here two hidden layers of 9 and 11 neurons are taken to process).To check the loss mean square error is uesd.Then the testing set is taken and fitted, at last the model is checked for accuracy via preiction.

## Neural Network Model

![](https://github.com/RanjithD18/basic-nn-model/blob/main/Screenshot%202023-08-16%20180210.png)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
~~~
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('dp').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'Input':'float'})
df = df.astype({'Output':'float'})
df.head()
X=df[['Input']].values
y=df[['Output']].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=33)

from sklearn.preprocessing import MinMaxScaler
Scaler=MinMaxScaler()
Scaler.fit(X_train)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
ai_brain=Sequential([
    Dense(9,activation='relu'),
    Dense(11,activation='relu'),
    
    Dense(1)
])
ai_brain.compile(
    optimizer = 'rmsprop',
    loss = 'mse'
)

ai_brain.fit(X_train1,y_train,epochs = 3000)

ai_brain.evaluate(X_test,y_test)

X_n1 = [[101]]

X_n1_1 = Scaler.transform(X_n1)

ai_brain.predict(X_n1_1)

~~~
## Dataset Information

![](https://github.com/RanjithD18/basic-nn-model/blob/main/Screenshot1.png)

## OUTPUT

### Training Loss Vs Iteration Plot

![](https://github.com/RanjithD18/basic-nn-model/blob/main/Screenshot2.png)

### Test Data Root Mean Squared Error

![](https://github.com/RanjithD18/basic-nn-model/blob/main/Screenshot3.png)

### New Sample Data Prediction

![](https://github.com/RanjithD18/basic-nn-model/blob/main/Screenshot4.png)

## RESULT
Thus a neural network regression model for the given dataset is written and executed successfully
