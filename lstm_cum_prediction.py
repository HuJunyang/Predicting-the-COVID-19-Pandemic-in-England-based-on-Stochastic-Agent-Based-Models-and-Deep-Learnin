# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 15:10:02 2021

fitting with England data
inputs: 'cumCasesBySpecimenDate'
outputs: 'cumCasesBySpecimenDate'

sequence size = 2,3,4
number of neurons =  16,24,32,40,48
test size = 21,28,35,42,49,56

@author: JunyangHu
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import seaborn as sns
import keras
from sklearn.preprocessing import MinMaxScaler #for data preprocessing
from keras.preprocessing.sequence import TimeseriesGenerator
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#data loading
##load data and parse index with date and rename the variable names for conveneience
df_cases = pd.read_csv('D:/dt/data/cases.csv',parse_dates=['date'],index_col="date")
df_deaths = pd.read_csv('D:/dt/data/deaths.csv',parse_dates=['date'],index_col="date")
df_tests = pd.read_csv('D:/dt/data/tests.csv',parse_dates=['date'],index_col="date")
df_hospitalized = pd.read_csv('D:/dt/data/patients_admitted_in_hospital.csv',parse_dates=['date'],index_col="date")
df_total = pd.concat([df_cases, df_deaths, df_tests, df_hospitalized], axis=1)
df_total = df_total.fillna(0)
df_raw = df_total['cumCasesBySpecimenDate']
df_raw = df_raw.sort_index()
df_raw = df_raw.astype(float) #select the data for analysing and change the data type to suit for normalizing
#df_raw.head()
df = df_raw.truncate(after='2020-12-03')#select data before 2020-12-03
df_dates = df.index

df.tail()

#general time series data visualization
##a general picture of the cases trend for daily cases and cumulated cases
fig0 = plt.figure(figsize=(10,6),dpi=500)
plt.style.use('seaborn')
sns.set_palette('muted',8)
plt.title('Cumulative hospitalisations in England')
plt.xticks(rotation=45)
plt.plot(df)
print("Total days in the dataset", len(df))




test_size = 21
n_future = 1
n_features = 1

seq_size = 2 ## 2 3 ##
batch_size = 128 ##16 64 128 150 256#
hidden_units = 32 ## 16 32 48 ##
activation = 'relu' ## 'relu' 'tanh' ##



train_size = len(df)-test_size
train_set = np.array(df.iloc[:train_size]).reshape(-1,1)
test_set = np.array(df.iloc[train_size:]).reshape(-1,1)
print(len(train_set),len(test_set))

#data preprocessing
##LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
## normalize the dataset
scaler = MinMaxScaler()
scaler.fit(train_set)
train_scaled = scaler.transform(train_set)
test_scaled = scaler.transform(test_set)
print(len(train_scaled))


# train_features = train_scaled.to_numpy().tolist()
# train_target = train_scaled['newCasesBySpecimenDate'].tolist()

## generate data in sequences (sliding window)
#Sequence size has an impact on prediction, especially since COVID is unpredictable!
seq_size = seq_size
n_features = n_features ## number of features. This dataset is univariate so it is 1

trainX = []
trainY = []




n_future = n_future  # Number of days we want to predict into the future
n_past = seq_size     # Number of past days we want to use to predict the future

for i in range(n_past, len(train_scaled) - n_future +1):
    trainX.append(train_scaled[i - n_past:i, 0:train_scaled.shape[1]])
    trainY.append(train_scaled[i + n_future - 1:i + n_future, 0])

trainX, trainY = np.array(trainX), np.array(trainY)


testX = []
testY = []

n_future = 1  # Number of days we want to predict into the future
n_past = seq_size     # Number of past days we want to use to predict the future

for i in range(n_past, len(test_scaled) - n_future +1):
    testX.append(test_scaled[i - n_past:i, 0:train_scaled.shape[1]])
    testY.append(test_scaled[i + n_future - 1:i + n_future, 0])

testX, testY = np.array(testX), np.array(testY)


################################ baselineScore
# df_baselineScore = pd.read_csv('baselineScore.csv')
# test_baseline_label = test_set[seq_size:]
# test_baseline_predict = test_set[seq_size-1:-1]
# baselineScore = math.sqrt(mean_squared_error(test_baseline_label, test_baseline_predict))
# print('Train Score: %.2f RMSE' % (baselineScore))

# baselineScore1 = mean_squared_error(test_baseline_label, test_baseline_predict)
# print('Train Score: %.2f MSE' % (baselineScore1))

# baselineScore2 = mean_absolute_error(test_baseline_label, test_baseline_predict)
# print('Test Score: %.2f MAE' % (baselineScore2))

# baselineScore3 = mean_absolute_percentage_error(test_baseline_label, test_baseline_predict)
# print('Test Score: %.5f MAPE' % (baselineScore3))

# base_dict = {'var':'cumulative hospitalisations','RMSE':baselineScore,'MSE':baselineScore1,'MAE':baselineScore2,'MAPE':baselineScore3}
# df_baselineScore = df_baselineScore.append(base_dict,ignore_index=True)
# df_baselineScore.to_csv('baselineScore.csv')


##########################
###building up the LSTM model
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation,BatchNormalization

#Define Model 
model = Sequential()
model.add(LSTM(hidden_units, activation=activation, return_sequences=False, input_shape=(seq_size, n_features)))
#model.add(Dropout(0))
# model.add(LSTM(16, activation='tanh', return_sequences=False))
# model.add(BatchNormalization())
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
print('Train...')
##########################

# fit model
history = model.fit(trainX,trainY, 
                              validation_data=(testX,testY), 
                              epochs=500, batch_size=batch_size,verbose=0)

##############################################
#model.save('lstm_1_cum_hospitalisations')
##load model lstm_1_cum

# from keras.models import load_model
# model = keras.models.load_model('lstm_1_cum')
##############################################
##record





#plot the training and validation accuracy and loss at each epoch
# fig1 = plt.figure(figsize=(10,6),dpi=500)
# plt.style.use('seaborn')
# sns.set_palette('muted',8)
# loss1 = history.history['loss']
# val_loss1 = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.figure(1)
# plt.plot(epochs, loss, label='Training loss')
# plt.plot(epochs, val_loss, label='Validation loss')
# plt.title('Training and validation loss',fontsize=18)
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# make predictions

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions back to prescaled values
#This is to compare with original input values
#SInce we used minmaxscaler we can now use scaler.inverse_transform
#to invert the transformation.

#Perform inverse transformation to rescale back to original range
#Since we used 5 variables for transform, the inverse expects same dimensions
#Therefore, let us copy our values 5 times and discard them after inverse transform

trainPredict = scaler.inverse_transform(trainPredict)[:,0]

train_labels = scaler.inverse_transform(train_scaled)[seq_size:,0]


testPredict = scaler.inverse_transform(testPredict)[:,0]

test_labels = scaler.inverse_transform(test_scaled)[seq_size:,0]

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(train_labels, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))

trainScore1 = mean_squared_error(train_labels, trainPredict)
print('Train Score: %.2f MSE' % (trainScore1))

testScore = math.sqrt(mean_squared_error(test_labels, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

testScore1 = mean_squared_error(test_labels, testPredict)
print('Test Score: %.2f MSE' % (testScore1))

testScore2 = mean_absolute_error(test_labels, testPredict)
print('Test Score: %.2f MAE' % (testScore2))

testScore3 = mean_absolute_percentage_error(test_labels, testPredict)
print('Test Score: %.5f MAPE' % (testScore3))


# ##########################################
# #forecasting error
# test_forecast = [] #Empty list to populate later with predictions

# current_batch_test = train_scaled[-seq_size:] #Final data points in train 
# current_batch_test = current_batch_test.reshape(1, seq_size, n_features) #Reshape

# ## Predict future, beyond test dates
# future = test_size #Days
# for i in range(future):
#     current_pred_test = model.predict(current_batch_test)[0]
#     test_forecast.append(current_pred_test)
#     current_batch_test = np.append(current_batch_test[:,1:,:],[[current_pred_test]],axis=1)

# ### Inverse transform to before scaling so we get actual numbers
# rescaled_test_forecast = scaler.inverse_transform(test_forecast)


# ##########################################


# shift train predictions for plotting
#we must shift the predictions so that they align on the x-axis with the original dataset. 
trainPredictPlot = np.empty_like(df.values)
trainPredictPlot[:, ] = np.nan
trainPredictPlot[seq_size:len(trainPredict)+seq_size] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(df.values)
testPredictPlot[:, ] = np.nan
testPredictPlot[len(trainPredict)+(seq_size*2):] = testPredict

# plot baseline and predictions
sns.set_palette('muted',8)
fig2 = plt.figure(figsize=(10,6),dpi=500)
plt.plot(df_dates,df.values,label='actual values')
plt.plot(df_dates,trainPredictPlot,label='training prediction values')
plt.plot(df_dates,testPredictPlot,label = 'validating prediction values')
plt.xticks(rotation=45)
plt.title('LSTM - Cumulative hospitalisations prediction',fontsize=18)
plt.legend()
plt.show()



# #forecast
# prediction = [] #Empty list to populate later with predictions

# current_batch = train_scaled[-seq_size:] #Final data points in train 
# current_batch = current_batch.reshape(1, seq_size, n_features) #Reshape

# ## Predict future, beyond test dates
# future = 28 #Days
# for i in range(len(test_set) + future):
#     current_pred = model.predict(current_batch)[0]
#     prediction.append(current_pred)
#     current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

# ### Inverse transform to before scaling so we get actual numbers
# rescaled_prediction = scaler.inverse_transform(prediction)

# time_series_array = df.iloc[train_size:].index  #Get dates for test data

# #Add new dates for the forecast period
# for k in range(0, future):
#     time_series_array = time_series_array.append(time_series_array[-1:] + pd.DateOffset(1))

# #Create a dataframe to capture the forecast data
# df_forecast = pd.DataFrame(columns=["actual_confirmed","predicted"], index=time_series_array)

# df_forecast.loc[:,"predicted"] = rescaled_prediction[:,0]
# df_forecast["actual_confirmed"] = df_raw[time_series_array]


#forecast
prediction = [] #Empty list to populate later with predictions

current_batch = test_scaled[-seq_size:] #Final data points in train 
current_batch = current_batch.reshape(1, seq_size, n_features) #Reshape

## Predict future, beyond test dates
future = 14 #Days
for i in range(future):
    current_pred = model.predict(current_batch)[0]
    prediction.append(current_pred)
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

### Inverse transform to before scaling so we get actual numbers
rescaled_prediction = scaler.inverse_transform(prediction)

time_series_array = df.iloc[train_size+seq_size:].index  #Get dates for test data

#Add new dates for the forecast period
for k in range(0, future):
    time_series_array = time_series_array.append(time_series_array[-1:] + pd.DateOffset(1))

#Create a dataframe to capture the forecast data
df_forecast = pd.DataFrame(columns=["actual confirmed","predicted values"], index=time_series_array)
df_forecast.loc[:'2020-12-03',"predicted values"] = testPredict
df_forecast.loc['2020-12-04':,"predicted values"] =  rescaled_prediction[:,0]
df_forecast.loc[:,"actual confirmed"] = df_raw[time_series_array]

df_forecast['1-order-d-actual']=df_forecast['actual confirmed'].diff(1)
df_forecast['1-order-d-pred']=df_forecast['predicted values'].diff(1)

#Plot
plt.style.use('seaborn')
sns.set_palette('muted',8)
fig3 = plt.figure(figsize=(10,6),dpi=500)
plt.title("LSTM - Cumulative hospitalisations forecasting",fontsize=18)
plt.plot(df_forecast.loc[:'2020-12-03','actual confirmed'], label='actual confirmed')
plt.plot(df_forecast.loc['2020-12-03':,'actual confirmed'],label='actual confirmed (future)',linestyle='--',alpha=0.5)
plt.plot(df_forecast['predicted values'], label='predicted values')
plt.axvline(x='2020-12-03',linestyle="--",c="r",alpha = 0.5)#添加垂直直线
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter(''))
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(7))
plt.xticks(rotation=15,fontsize=10)
plt.xlabel('')
plt.ylabel('Number of cumulative hospitalisations',fontsize=12)
plt.tick_params(labelsize=10)
plt.legend()
plt.axvspan('2020-12-03','2020-12-17' , facecolor='red', alpha=0.05)
#plt.savefig(dpi=1000,figsize=(6,4),fname='lstm_1_cum_28_days_pred')
plt.show()

fig4 = plt.figure(figsize=(10,6),dpi=500)
plt.title("LSTM - New daily hospitalisations (First-order difference) forecasting",fontsize=18)
plt.plot(df_forecast.loc[:'2020-12-03','1-order-d-actual'], label='actual confirmed (First-order difference)')
plt.plot(df_forecast.loc['2020-12-03':,'1-order-d-actual'],label='actual confirmed (First-order difference) (future)',linestyle='--',alpha=0.5)
plt.plot(df_forecast['1-order-d-pred'], label='predicted values (First-order difference)')
plt.axvline(x='2020-12-03',linestyle="--",c="r",alpha = 0.5)#添加垂直直线
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter(''))
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(7))
plt.xticks(rotation=15,fontsize=10)
plt.xlabel('')
plt.ylabel('Number of new daily hospitalisations (First-ordder difference)',fontsize=12)
plt.tick_params(labelsize=10)
plt.legend()
plt.axvspan('2020-12-03','2020-12-17' , facecolor='red', alpha=0.05)
plt.show()

# df_forecast_error = pd.read_csv('forecasting error.csv')
# df_forecast['lstm_cases-1-d'] = abs(df_forecast['1-order-d-pred']-df_forecast['1-order-d-actual'])/df_forecast['1-order-d-actual']
# df_forecast_error['lstm_cases-1-d'] = df_forecast['lstm_cases-1-d'][1:].values
# df_forecast_error.to_csv('forecasting error.csv')
# # #######################################################
# # #sensitivity analysis
# sen_dict = {'model':'lstm_cum_prediction','seq_size':seq_size,'batch_size':batch_size, 
#             'hidden_units':hidden_units,'activation':activation,'test errors':testScore}

# df_sensitivity=df_sensitivity.append(sen_dict,ignore_index=True)

