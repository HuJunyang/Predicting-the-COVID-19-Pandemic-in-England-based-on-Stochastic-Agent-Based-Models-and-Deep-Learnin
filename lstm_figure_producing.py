# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 20:43:33 2021

@author: JunyangHu
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%#############################################
#autocorrelation tests
df_cases = pd.read_csv('D:/dt/data/cases.csv',parse_dates=['date'],index_col="date")
df_deaths = pd.read_csv('D:/dt/data/deaths.csv',parse_dates=['date'],index_col="date")
df_tests = pd.read_csv('D:/dt/data/tests.csv',parse_dates=['date'],index_col="date")
df_hospitalized = pd.read_csv('D:/dt/data/patients_admitted_in_hospital.csv',parse_dates=['date'],index_col="date")
df_total = pd.concat([df_cases, df_deaths, df_tests, df_hospitalized], axis=1)
df_total = df_total.fillna(0)
df_new = df_total['newCasesBySpecimenDate'].sort_index().astype(float).truncate(after='2020-12-03')
df_cum = df_total['cumCasesBySpecimenDate'].sort_index().astype(float).truncate(after='2020-12-03')
df_new_deaths = df_total['newDeaths28DaysByDeathDate'].sort_index().astype(float).truncate(after='2020-12-03')
df_cum_deaths = df_total['cumDeaths28DaysByDeathDate'].sort_index().astype(float).truncate(after='2020-12-03')
df_new_hospitalisations = df_total['newAdmissions'].sort_index().astype(float).truncate(after='2020-12-03')
df_cum_hospitalisations = df_total['cumAdmissions'].sort_index().astype(float).truncate(after='2020-12-03')



from statsmodels.stats.diagnostic import acorr_ljungbox as lb_test
lb_test_cum = lb_test(df_cum,lags=40,return_df=True)
lb_test_new = lb_test(df_new,lags=40,return_df=True)
lb_test_cum_deaths = lb_test(df_cum_deaths,lags=40,return_df=True)
lb_test_new_deaths = lb_test(df_new_deaths,lags=40,return_df=True)
lb_test_cum_hospitalisations = lb_test(df_cum_hospitalisations,lags=40,return_df=True)
lb_test_new_hospitalisations = lb_test(df_new_hospitalisations,lags=40,return_df=True)
fig = plt.figure(figsize=(10,6),dpi=500)
plt.plot(lb_test_cum['lb_pvalue'],label='cumulative confirmed cases', marker = 'v',alpha=1)
plt.plot(lb_test_new['lb_pvalue'],label='new daily confirmed cases',marker = 's',alpha=0.9)
plt.plot(lb_test_cum_deaths['lb_pvalue'],label='cumulative confirmed deaths',marker = 'D',alpha=0.8)
plt.plot(lb_test_new_deaths['lb_pvalue'],label='new daily confirmed deaths',marker='H',alpha=0.7)
plt.plot(lb_test_cum_hospitalisations['lb_pvalue'],label='cumulative hospitalisations',marker = 'o',alpha=0.6)
plt.plot(lb_test_new_hospitalisations['lb_pvalue'],label='new daily hospitalisations',marker = 'p',alpha=0.5)

plt.legend()
plt.title('Ljung-Box test')
plt.ylabel('p-value')
plt.xlabel('lags')
plt.show()

from pandas.plotting import autocorrelation_plot
plt.figure(figsize=(10,6),dpi=500)
plt.style.use('seaborn')
sns.set_palette('muted',8)
autocorrelation_plot(df_new,label='new daily confirmed cases')
autocorrelation_plot(df_cum,label='cumulative confirmed cases')
autocorrelation_plot(df_new_deaths,label='new daily confirmed deaths')
autocorrelation_plot(df_cum_deaths,label='cumulative confirmed deaths')
autocorrelation_plot(df_new_hospitalisations,label='new daily hospitalisations')
autocorrelation_plot(df_cum_hospitalisations,label='cumulative hospitalisations')
plt.legend()
plt.title('ACF plot')
plt.show()

#%%
##############################################
##heatmap
df_para = pd.read_csv('D:/dt/data/lstm_cum_hyperparameters_tunning.csv')
df_para.head()



df_seq_size_2 = df_para[df_para['seq_size']==2][['test_size','units_1','test_error_rmse']]
df_seq_size_2 = df_seq_size_2.pivot_table(values='test_error_rmse',index='test_size',columns='units_1')

df_seq_size_3 = df_para[df_para['seq_size']==3][['test_size','units_1','test_error_rmse']]
df_seq_size_3 = df_seq_size_3.pivot_table(values='test_error_rmse',index='test_size',columns='units_1')

df_seq_size_4 = df_para[df_para['seq_size']==4][['test_size','units_1','test_error_rmse']]
df_seq_size_4 = df_seq_size_4.pivot_table(values='test_error_rmse',index='test_size',columns='units_1')


plt.figure(figsize=(18,6),dpi=500)
plt.style.use('ggplot')
sns.set_palette('muted',8)
plt.subplot(1, 3, 1)
heatmap1 = sns.heatmap(data=df_seq_size_2,cmap="RdBu_r",square = True, linewidths=0.3, vmin=5000,vmax=500000)
heatmap1.set_title('Sequence size = 2',fontsize = 14)
heatmap1.set_xlabel('Number of neurons in the layer')
heatmap1.set_ylabel('Testing data size')

plt.subplot(1, 3, 2)
heatmap2 = sns.heatmap(data=df_seq_size_3,cmap="RdBu_r",square = True,linewidths=0.3,vmin=5000,vmax=500000)
heatmap2.set_title('Sequence size = 3',fontsize = 14)
heatmap2.set_xlabel('Number of neurons in the layer')
heatmap2.set_ylabel('Testing data size')

plt.subplot(1, 3, 3)
heatmap3 = sns.heatmap(data=df_seq_size_4,cmap="RdBu_r",square = True,linewidths=0.3,vmin=5000,vmax=500000)
heatmap3.set_title('Sequence size = 4',fontsize = 14)
heatmap3.set_xlabel('Number of neurons in the layer')
heatmap3.set_ylabel('Testing data size')

plt.suptitle('Testing errors',fontsize=18)

plt.show()


plt.set_title('Test errors')
plt.set_xlabel('Number of neurons in the layer')
plt.set_ylabel('Testing data size')
plt.show()




#################################################
##loss plot
fig1 = plt.figure(figsize=(10,6),dpi=500)


epochs = range(1, len(loss1) + 1)
plt.plot(epochs, loss1,  label='Training loss-lstm-cumulated cases',linewidth=0.8)
plt.plot(epochs, val_loss1,  label='Validation loss-lstm-cumulated cases',linewidth=0.8)


plt.plot(epochs, loss2,  label='Training loss-gru-cumulated cases',linewidth=0.8)
plt.plot(epochs, val_loss2,  label='Validation loss-gru-cumulated cases',linewidth=0.8)



plt.plot(epochs, loss3,  label='Training loss-lstm-new daily cases',linewidth=0.8)
plt.plot(epochs, val_loss3,  label='Validation loss-lstm-new daily cases',linewidth=0.8)


plt.plot(epochs, loss4,  label='Training loss-gru-new daily cases',linewidth=0.8)
plt.plot(epochs, val_loss4,  label='Validation loss-gru-new daily cases',linewidth=0.8)



plt.plot(epochs, loss5,  label='Training loss-lstm-new daily cases (3 days moving average)',linewidth=0.8)
plt.plot(epochs, val_loss5,  label='Validation loss-lstm-new daily cases (3 days moving average)',linewidth=0.8)


plt.plot(epochs, loss6,  label='Training loss-gru-new daily cases (3 days moving average)',linewidth=0.8)
plt.plot(epochs, val_loss6,  label='Validation loss-gru-new daily cases (3 days moving average)',linewidth=0.8)



plt.plot(epochs, loss7,  label='Training loss-lstm-new daily cases (5 days moving average)',linewidth=0.8)
plt.plot(epochs, val_loss7,  label='Validation loss-lstm-new daily cases (5 days moving average)',linewidth=0.8)


plt.plot(epochs, loss8,  label='Training loss-gru-new daily cases (5 days moving average)',linewidth=0.8)
plt.plot(epochs, val_loss8,  label='Validation loss-gru-new daily cases (5 days moving average)',linewidth=0.8)

plt.title('Training and validation loss',fontsize=18)

#plt.ylim(ymax = 0.5,ymin=0)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


loss_plot_dict = {'loss1':loss1,'loss2':loss2,'loss3':loss3,'loss4':loss4,'loss5':loss5,'loss6':loss6,'loss7':loss7,
                  'loss8':loss8,'val_loss1':val_loss1,'val_loss2':val_loss2,'val_loss3':val_loss3,'val_loss4':val_loss4,
                  'val_loss5':val_loss5,'val_loss6':val_loss6,'val_loss7':val_loss7,'val_loss8':val_loss8}
df_loss_plot = pd.DataFrame(loss_plot_dict)
df_loss_plot.to_excel('loss plot.xlsx')
###########
#%%
###############################################
##sensitivity analysis
df_sensitivity_gru_new_5 = df_sensitivity_gru_new_5.drop(index=16)
opt = {'model':'gru_new_prediction','seq_size':3,'batch_size':128, 
            'hidden_units':128,'activation':'relu','test errors':995.43}

df_sensitivity_gru_new_5 = df_sensitivity_gru_new_5.append(opt,ignore_index=True)

fig1 = plt.figure(figsize=(24,4),dpi=500)
plt.ylabel('RMSE')
plt.subplot(1, 4, 1)
box1=df_sensitivity[df_sensitivity['seq_size']==2]['test errors']
box2=df_sensitivity[df_sensitivity['seq_size']==3]['test errors']
plt.style.use('seaborn')
sns.set_palette('muted',8)
plt.boxplot([box1,box2],labels=['2','3'])
plt.title('sequence size')

plt.subplot(1, 4, 2)
box3=df_sensitivity[df_sensitivity['activation']=='relu']['test errors']
box4=df_sensitivity[df_sensitivity['activation']=='tanh']['test errors']
plt.boxplot([box3,box4],labels=['relu','tanh'])
plt.title('activation function')

plt.subplot(1, 4, 3)
box5=df_sensitivity[df_sensitivity['batch_size']==16]['test errors']
box6=df_sensitivity[df_sensitivity['batch_size']==64]['test errors']
box7=df_sensitivity[df_sensitivity['batch_size']==128]['test errors']
box8=df_sensitivity[df_sensitivity['batch_size']==150]['test errors']
box9=df_sensitivity[df_sensitivity['batch_size']==256]['test errors']
plt.boxplot([box5,box6,box7,box8,box9],labels=['16','64','128', '150','256'])
plt.title('batch size')

plt.subplot(1, 4, 4)
box10=df_sensitivity[df_sensitivity['hidden_units']==16]['test errors']
box11=df_sensitivity[df_sensitivity['hidden_units']==32]['test errors']
box12=df_sensitivity[df_sensitivity['hidden_units']==48]['test errors']
plt.boxplot([box10,box11,box12],labels=['16','32','48'])
plt.title('number of hidden units')

plt.suptitle('LSTM - cumulative confirmed cases',fontsize=14)
plt.show()



fig2 = plt.figure(figsize=(24,4),dpi=500)
plt.ylabel('RMSE')
plt.subplot(1, 4, 1)
box1=df_sensitivity_lstm_deaths[df_sensitivity_lstm_deaths['seq_size']==2]['test errors']
box2=df_sensitivity_lstm_deaths[df_sensitivity_lstm_deaths['seq_size']==3]['test errors']
plt.style.use('seaborn')
sns.set_palette('muted',8)
plt.boxplot([box1,box2],labels=['2','3'])
plt.title('sequence size')

plt.subplot(1, 4, 2)
box3=df_sensitivity_lstm_deaths[df_sensitivity_lstm_deaths['activation']=='relu']['test errors']
box4=df_sensitivity_lstm_deaths[df_sensitivity_lstm_deaths['activation']=='tanh']['test errors']
plt.boxplot([box3,box4],labels=['relu','tanh'])
plt.title('activation function')

plt.subplot(1, 4, 3)
box5=df_sensitivity_lstm_deaths[df_sensitivity_lstm_deaths['batch_size']==64]['test errors']
box6=df_sensitivity_lstm_deaths[df_sensitivity_lstm_deaths['batch_size']==128]['test errors']
box7=df_sensitivity_lstm_deaths[df_sensitivity_lstm_deaths['batch_size']==256]['test errors']
plt.boxplot([box5,box6,box7],labels=['64','128','256'])
plt.title('batch size')

plt.subplot(1, 4, 4)
box8=df_sensitivity_lstm_deaths[df_sensitivity_lstm_deaths['hidden_units']==48]['test errors']
box9=df_sensitivity_lstm_deaths[df_sensitivity_lstm_deaths['hidden_units']==96]['test errors']
box10=df_sensitivity_lstm_deaths[df_sensitivity_lstm_deaths['hidden_units']==128]['test errors']
box11=df_sensitivity_lstm_deaths[df_sensitivity_lstm_deaths['hidden_units']==256]['test errors']
plt.boxplot([box8,box9,box10,box11],labels=['48','96','128','256'])
plt.title('number of hidden units')

plt.suptitle('LSTM - new daily confirmed deaths',fontsize=14)
plt.show()

#%%
import matplotlib.ticker as mtick

plt.style.use('seaborn')
sns.set_palette('muted',8)
xaxis = ['2020-12-04', '2020-12-05', '2020-12-06', '2020-12-07', '2020-12-08', '2020-12-09', '2020-12-10', '2020-12-11', '2020-12-12', '2020-12-13', '2020-12-14', '2020-12-15', '2020-12-16', '2020-12-17']
x1 = range(len(xaxis))
x2 = [i+0.4 for i in x1]
fig4 = plt.figure(figsize=(10,6),dpi=500)
bar1 = plt.bar(x1, width=0.5, height = df_forecast_error.loc[18:,'lstm_new_hospitalisations_3'].values, label = 'LSTM-New daily hospitalisations forecasting')
bar2 = plt.bar(x2, width=0.5, height = df_forecast_error.loc[18:,'gru_new_hospitalisations_3'].values, label = 'GRU-New daily hospitalisations forecasting')
plt.xticks(x1,xaxis)
plt.xticks(rotation=30,fontsize=12)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.ylabel('percentage error')
plt.title('New daily hospitalisations forecasting percentage error bar chart (3 days moving average) ',fontsize=14)
plt.legend()
plt.show()