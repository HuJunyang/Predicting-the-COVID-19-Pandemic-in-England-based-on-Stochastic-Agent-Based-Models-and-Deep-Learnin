# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 16:29:34 2021

@author: JunyangHu
"""

import covasim as cv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np

a = cv.load('results_Vac/eng.a.obj')

s1 = cv.load('results_Vac/eng.1.obj')
s2 = cv.load('results_Vac/eng.2.obj')
s3 = cv.load('results_Vac/eng.3.obj')
s4 = cv.load('results_Vac/eng.4.obj')
s5 = cv.load('results_Vac/eng.5.obj')
s6 = cv.load('results_Vac/eng.6.obj')
s7 = cv.load('results_Vac/eng.7.obj')
s8 = cv.load('results_Vac/eng.8.obj')
s9 = cv.load('results_Vac/eng.9.obj')
s10 = cv.load('results_Vac/eng.10.obj')
s11 = cv.load('results_Vac/eng.11.obj')
s12 = cv.load('results_Vac/eng.12.obj')

help(cv.Sim.plot)
#%% calibration plot
to_plot = {
    'Cumulative confirmed cases': ['cum_diagnoses'],
    #'Cumulative hospitalisations': ['cum_severe'],
    #'Cumulative deaths': ['cum_deaths'],
    'New daily confirmed cases': ['new_diagnoses'],
    #'New daily hospitalisations': ['new_severe'],
    #'New daily deaths': ['new_deaths'],
    'Effective reproduction number R': ['r_eff']
}
cv.options.set(font_size=12,dpi=500)
plt.style.use('seaborn')
sns.set_palette('muted',8)
s12.plot(to_plot=to_plot, do_save=True, do_show=False, fig_path='England_calibration.png',
                      legend_args={'loc': 'upper left'}, axis_args={'hspace': 0.2,'wspace':0.1}, interval=50, n_cols=3,
                      fig_args=dict(figsize=(25,5)),start_day='2020-12-04',end_day='2021-2-04',grid=1)

s1.plot('variants', do_save=True, do_show=False,axis_args={'hspace': 0.2,'wspace':0.1}, interval=50, n_cols=1,
                      fig_args=dict(figsize=(6,16)),start_day='2020-8-20',end_day='2021-2-04',grid=1)

# fa = s1.plot(to_plot={'New daily confirmed cases (Scenario 1)':['cum_vaccinated']},start_day='2020-12-18',end_day='2021-2-28',
#               interval=7, grid=1, fig_args=dict(figsize=(10,6)))


#%% 12 scenarios daily cases plot

fc1 = s1.plot(to_plot={'New daily confirmed cases (Scenario 1)':['new_diagnoses']},start_day='2020-12-18',end_day='2021-2-28',
              interval=7, grid=1, fig_args=dict(figsize=(10,6)))


fc2 = s2.plot(to_plot={'New daily confirmed cases (Scenario 2)':['new_diagnoses']},start_day='2020-12-18',end_day='2021-2-28',
              interval=7, grid=1, fig_args=dict(figsize=(10,6)))

fc3 = s3.plot(to_plot={'New daily confirmed cases (Scenario 3)':['new_diagnoses']},start_day='2020-12-18',end_day='2021-2-28',
              interval=7, grid=1, fig_args=dict(figsize=(10,6)))


fc4 = s4.plot(to_plot={'New daily confirmed cases (Scenario 4)':['new_diagnoses']},start_day='2020-12-18',end_day='2021-2-28',
              interval=7, grid=1, fig_args=dict(figsize=(10,6)))

fc5 = s5.plot(to_plot={'New daily confirmed cases (Scenario 5)':['new_diagnoses']},start_day='2020-12-18',end_day='2021-2-28',
              interval=7, grid=1, fig_args=dict(figsize=(10,6)))

fc6 = s6.plot(to_plot={'New daily confirmed cases (Scenario 6)':['new_diagnoses']},start_day='2020-12-18',end_day='2021-2-28',
              interval=7, grid=1, fig_args=dict(figsize=(10,6)))

fc7 = s7.plot(to_plot={'New daily confirmed cases (Scenario 7)':['new_diagnoses']},start_day='2020-12-18',end_day='2021-2-28',
              interval=7, grid=1, fig_args=dict(figsize=(10,6)))

fc8 = s8.plot(to_plot={'New daily confirmed cases (Scenario 8)':['new_diagnoses']},start_day='2020-12-18',end_day='2021-2-28',
              interval=7, grid=1, fig_args=dict(figsize=(10,6)))

fc9 = s9.plot(to_plot={'New daily confirmed cases (Scenario 9)':['new_diagnoses']},start_day='2020-12-18',end_day='2021-2-28',
              interval=7, grid=1, fig_args=dict(figsize=(10,6)))

fc10 = s10.plot(to_plot={'New daily confirmed cases (Scenario 10)':['new_diagnoses']},start_day='2020-12-18',end_day='2021-2-28',
              interval=7, grid=1, fig_args=dict(figsize=(10,6)))

fc11 = s11.plot(to_plot={'New daily confirmed cases (Scenario 11)':['new_diagnoses']},start_day='2020-12-18',end_day='2021-2-28',
              interval=7, grid=1, fig_args=dict(figsize=(10,6)))

fc12 = s12.plot(to_plot={'New daily confirmed cases (Scenario 12)':['new_diagnoses']},start_day='2020-12-18',end_day='2021-2-28',
              interval=7, grid=1, fig_args=dict(figsize=(10,6)))

df_cases = pd.read_excel('England_Covid_cases_feb28.xlsx')

scpd = pd.DataFrame({'date':pd.to_datetime(s1.results['date'],format='%Y-%m-%d'),
                     's1_new_cases':s1.results['new_diagnoses'],
                     's2_new_cases':s2.results['new_diagnoses'],
                     's3_new_cases':s3.results['new_diagnoses'],
                     's4_new_cases':s4.results['new_diagnoses'],
                     's5_new_cases':s5.results['new_diagnoses'],
                     's6_new_cases':s6.results['new_diagnoses'],
                     's7_new_cases':s7.results['new_diagnoses'],
                     's8_new_cases':s8.results['new_diagnoses'],
                     's9_new_cases':s9.results['new_diagnoses'],
                     's10_new_cases':s10.results['new_diagnoses'],
                     's11_new_cases':s11.results['new_diagnoses'],
                     's12_new_cases':s12.results['new_diagnoses'],
                     'actual_new_cases':df_cases['new_diagnoses']
                     })

scpd_p = scpd.set_index('date',inplace=False)['2020-12-04':'2021-2-04']


######### daily cases in two months
f_12_nc = plt.figure(figsize=(10,6),dpi=500)
plt.style.use('seaborn')
# sns.set_palette('muted',13)
plt.plot(scpd_p['s1_new_cases'],label = 'scenario 1',alpha=0.5,c='#26f7fd')
plt.plot(scpd_p['s2_new_cases'],label = 'scenario 2',alpha=0.5,c='#02d8e9')
plt.plot(scpd_p['s3_new_cases'],label = 'scenario 3',alpha=0.5,c='#06b1c4')
plt.plot(scpd_p['s4_new_cases'],label = 'scenario 4',alpha=0.5,c='#047495')
plt.plot(scpd_p['s5_new_cases'],label = 'scenario 5',alpha=0.5,c='#FFE4C4')
plt.plot(scpd_p['s6_new_cases'],label = 'scenario 6',alpha=0.5,c='#FFDEAD')
plt.plot(scpd_p['s7_new_cases'],label = 'scenario 7',alpha=0.5,c='#FFA500')
plt.plot(scpd_p['s8_new_cases'],label = 'scenario 8',alpha=0.5,c='#FF8C00')
plt.plot(scpd_p['s9_new_cases'],label = 'scenario 9',alpha=0.5,c='#d6b4fc')
plt.plot(scpd_p['s10_new_cases'],label = 'scenario 10',alpha=0.5,c='#c48efd')
plt.plot(scpd_p['s11_new_cases'],label = 'scenario 11',alpha=0.5,c='#a55af4')
plt.plot(scpd_p['s12_new_cases'],label = 'scenario 12',alpha=0.5,c='#9900fa')
plt.scatter(x=scpd_p.index,y=scpd_p['actual_new_cases'],label = 'actual daily cases',marker='o',c='#ff796c',alpha=0.8)

plt.legend()
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(7))
plt.xticks(rotation=15,fontsize=10)
plt.title('New daily cases long-term forecasting')
plt.show()



########## cumulated cases in two months
sum1 = sum(scpd_p['s1_new_cases'])
sum2 = sum(scpd_p['s2_new_cases'])
sum3 = sum(scpd_p['s3_new_cases'])
sum4 = sum(scpd_p['s4_new_cases'])
sum5 = sum(scpd_p['s5_new_cases'])
sum6 = sum(scpd_p['s6_new_cases'])
sum7 = sum(scpd_p['s7_new_cases'])
sum8 = sum(scpd_p['s8_new_cases'])
sum9 = sum(scpd_p['s9_new_cases'])
sum10 = sum(scpd_p['s10_new_cases'])
sum11 = sum(scpd_p['s11_new_cases'])
sum12 = sum(scpd_p['s12_new_cases'])
sum_a = sum(scpd_p['actual_new_cases'])

f_12_nc = plt.figure(figsize=(10,6),dpi=500)
plt.style.use('seaborn')
plt.bar(x='scenario 1',height=sum1,alpha=0.7,color='#26f7fd')
plt.bar(x='scenario 2',height=sum2,alpha=0.7,color='#02d8e9')
plt.bar(x='scenario 3',height=sum3,alpha=0.7,color='#06b1c4')
plt.bar(x='scenario 4',height=sum4,alpha=0.7,color='#047495')
plt.bar(x='scenario 5',height=sum5,alpha=0.7,color='#FFE4C4')
plt.bar(x='scenario 6',height=sum6,alpha=0.7,color='#FFDEAD')
plt.bar(x='scenario 7',height=sum7,alpha=0.7,color='#FFA500')
plt.bar(x='scenario 8',height=sum8,alpha=0.7,color='#FF8C00')
plt.bar(x='scenario 9',height=sum9,alpha=0.7,color='#d6b4fc')
plt.bar(x='scenario 10',height=sum10,alpha=0.7,color='#c48efd')
plt.bar(x='scenario 11',height=sum11,alpha=0.7,color='#a55af4')
plt.bar(x='scenario 12',height=sum12,alpha=0.7,color='#9900fa')
plt.axhline(y=sum_a,linestyle="--",c='#ff796c',alpha = 0.5)
plt.ylim(ymin = 1000000)
plt.xticks(rotation=15)
plt.title('Cumulative cases in the long-term forecasting period')
plt.show()

s1.results['r_eff']

#%% 12 scenarios 
srpd = pd.DataFrame({'date':pd.to_datetime(s1.results['date'],format='%Y-%m-%d'),
                     's1_r_eff':s1.results['r_eff'],
                     's2_r_eff':s2.results['r_eff'],
                     's3_r_eff':s3.results['r_eff'],
                     's4_r_eff':s4.results['r_eff'],
                     's5_r_eff':s5.results['r_eff'],
                     's6_r_eff':s6.results['r_eff'],
                     's7_r_eff':s7.results['r_eff'],
                     's8_r_eff':s8.results['r_eff'],
                     's9_r_eff':s9.results['r_eff'],
                     's10_r_eff':s10.results['r_eff'],
                     's11_r_eff':s11.results['r_eff'],
                     's12_r_eff':s12.results['r_eff'],
                     })

srpd_p = srpd.set_index('date',inplace=False)['2020-12-04':'2021-2-04']

############# R from NHS
date = ['2020-12-04','2020-12-11','2020-12-18','2020-12-23','2021-01-08','2021-01-15','2021-01-22','2021-01-29',
        '2021-02-04']
upper = [1, 1, 1.3, 1.4, 1.4, 1.3, 1.0, 1.0, 0.9]
lower = [0.8, 0.8, 1.1, 1.1, 1.1, 1.1, 0.8, 0.7, 0.7]
r_eng = {'date':pd.to_datetime(date,format='%Y-%m-%d'), 'lower' : lower, 'upper':upper }
r_nhs = pd.DataFrame(r_eng).set_index('date',inplace=False)

####### plot
f_12_nc = plt.figure(figsize=(10,6),dpi=500)
plt.style.use('seaborn')
# sns.set_palette('muted',13)
plt.plot(srpd_p['s1_r_eff'],label = 'scenario 1',alpha=0.5,c='#26f7fd')
plt.plot(srpd_p['s2_r_eff'],label = 'scenario 2',alpha=0.5,c='#02d8e9')
plt.plot(srpd_p['s3_r_eff'],label = 'scenario 3',alpha=0.5,c='#06b1c4')
plt.plot(srpd_p['s4_r_eff'],label = 'scenario 4',alpha=0.5,c='#047495')
plt.plot(srpd_p['s5_r_eff'],label = 'scenario 5',alpha=0.5,c='#FFE4C4')
plt.plot(srpd_p['s6_r_eff'],label = 'scenario 6',alpha=0.5,c='#FFDEAD')
plt.plot(srpd_p['s7_r_eff'],label = 'scenario 7',alpha=0.5,c='#FFA500')
plt.plot(srpd_p['s8_r_eff'],label = 'scenario 8',alpha=0.5,c='#FF8C00')
plt.plot(srpd_p['s9_r_eff'],label = 'scenario 9',alpha=0.5,c='#d6b4fc')
plt.plot(srpd_p['s10_r_eff'],label = 'scenario 10',alpha=0.5,c='#c48efd')
plt.plot(srpd_p['s11_r_eff'],label = 'scenario 11',alpha=0.5,c='#a55af4')
plt.plot(srpd_p['s12_r_eff'],label = 'scenario 12',alpha=0.5,c='#9900fa')
plt.plot(r_nhs['lower'],label='lower bound (NHS)',alpha=0.8,c='#ff796c',linestyle="--")
plt.plot(r_nhs['upper'],label='upper bound (NHS)',alpha=0.8,c='#ff796c',linestyle="--")

#plt.axhline(y=1,linestyle="--",c='#ff796c',alpha = 0.5)
plt.legend()
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(7))
plt.xticks(rotation=15,fontsize=10)
plt.title('Forecasted effective reproduction number R')
plt.show()





############ boxplot
b1=srpd_p['s1_r_eff']
b2=srpd_p['s2_r_eff']
b3=srpd_p['s3_r_eff']
b4=srpd_p['s4_r_eff']
b5=srpd_p['s5_r_eff']
b6=srpd_p['s6_r_eff']
b7=srpd_p['s7_r_eff']
b8=srpd_p['s8_r_eff']
b9=srpd_p['s9_r_eff']
b10=srpd_p['s10_r_eff']
b11=srpd_p['s11_r_eff']
b12=srpd_p['s12_r_eff']

labels = ['scenario 1','scenario 2','scenario 3','scenario 4','scenario 5','scenario 6','scenario 7','scenario 8',
'scenario 9','scenario 10','scenario 11','scenario 12']

plt.boxplot([b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12],labels=labels)




