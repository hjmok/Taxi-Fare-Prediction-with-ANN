# Taxi-Fare-Prediction-with-ANN

in this project, an Artificial Neural Network is modeled using PyTorch to estimate a New York Taxi Fare amount and class.

## Dataset and Library
PyTorch was used as the main module for this model.
The dataset contains 120000 rows including information on the pickup time, latitude/longitude coordinates, and number of passengers. To get the full dataset, please see the following Kaggle link:

https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data


## Feature Engineering
One of the key columns for estimating taxi fare is the distance travelled. However, this information is not directly provided in the dataset. Instead the pickup and dropoff latitude and longitude were given as 4 separate columns. As such, the Haversine formula, which calculates distance across a sphere (Earth in this case) was utilized to feature engineer the distance with the given coordinates.

The datetime can also be useful information for predicting taxi fare. However, the Neural Network cannot be fed the date time string as data. Instead, the datetime column was further split into the time of day and day of the week, then One Hot Encoded such that the encoded value can be fed into the neural network.

## Data Preprocessing
The data had to be separated from categorical and continuous values. Continuous values included the latitude and longitude coordinates, passenger count, and distance travelled, which could directly be converted into tensors to act as inputs into the ANN.

However, categorical columns need to be encoded to be used as input for the ANN. This includes the the hour, AMorPM, and day of the week. As such, these columns were all encoded so that they had an associated numerical value, which can then be converted into a tensor to be fed into the ANN.

## ANN Model
The inputs for both the fare class and fare amount prediction models were the same, which were the coordinates, distance, date time values, and passenger count. The hidden layers went in the order of 256, 128, and 64 neurons, utilizing the Rectified Linear Unit Activation function. In addition, batch normalization and dropout layers were added to improve the model and reduce overfitting.

For the fare class model, output size was 2 since the only can only predict whether the Class is 1 for greater than $10 or 0 for less than $10. As such, the loss metric chosen was CrossEntropyLoss since this is a classification problem. In contrast, the fare amount model had an output size of 1 and loss metric of Root Mean Squared Error, since it is predicting a continuous dollar value. Both models used the Adam optimizer. 

## Results
Both models were trained on 450 epochs. As a result, the average root means squared error loss became 3.8764 for the Fare Amount Regression model. This means for every estimated fare, the model will be off by $3.88 on average. Moreover, the Fare Class Classification model resulted in a final the cross entropy loss of 0.272.

To improve both models, more optimization with the layer size and dropout rate can be tuned. In addition, the MSE LOSS plot below shows the loss was still declining slowly. As such, the Fare Amount Regression model could've kept training for a few more epochs to reduce the loss.

For the resulting image, please visit: https://hjmok.github.io/josephmok_portfolio/#/Taxi
