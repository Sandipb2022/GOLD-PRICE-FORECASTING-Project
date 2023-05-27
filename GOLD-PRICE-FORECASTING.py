#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Gold Price Forecast #(Forecasting Model)


# In[2]:


#1) Importing Libraries


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from datetime import datetime
import statsmodels.api as sm

 # Supressing warnings

import warnings             
warnings.filterwarnings('ignore')


# In[4]:


#2) Import Dataset


# In[5]:


#Load the data
df = pd.read_csv("Gold_data.csv",
                 parse_dates = ['date'],
                )


# In[6]:


df


# In[7]:


#We have imported the Dataset for Forecasting Model - Gold Price using Pandas.

#This Data Set contains 2182 Rows and 2 Columns

#Date Range - From 01/01/2016 To 21/12/2021. We Have total 5 year data.


# In[8]:


#3)EDA (Data Pre-Processing)


# In[9]:


df['date'] = pd.to_datetime(df['date'])
df


# In[10]:


#Basic Information
df.info()


# In[11]:


df.shape


# In[12]:


# Describe the data - Descriptive statistics.
df.describe()


# In[13]:


df.median()


# In[14]:


df.mode()


# In[15]:


# Finding Duplicated values / Null Values 


# In[16]:


# Null values

df.isnull().sum()


# In[17]:


# Duplicate Values

df.duplicated().sum()


# In[18]:


#It shows there is no Null and Duplicated values in the dataset


# In[19]:


# Know the datatypes
df.dtypes


# In[20]:


# Unique Values in the data
df['date'].unique()


# In[21]:


df['price'].unique()


# In[22]:


# Correlation 
df.corr()


# In[23]:


#Visualization of Data


# In[24]:


#Line Plot
df_eda = df.copy()
df_eda.set_index('date', inplace=True)


# In[25]:


# line plot
plt.figure(figsize=(20, 6))
sns.lineplot(y='price', x='date', data=df);
plt.title('Gold Prices vs Time');
plt.xlabel('Date');
plt.ylabel('Gold Price');


# In[26]:


#Histogram
plt.hist(df['price'],color='red')


# In[27]:


#From above visual trend we can see that the data has different trend at Different levels hence it is Non-Stationary


# In[28]:


### you can create a box plot for any numerical column using a single line of code.
box=df.boxplot(figsize=(8,8))


# In[29]:


#No outlier in the dataset


# In[30]:


#Displot To check Normality in the data
sns.distplot(df['price'])
plt.axvline(x=np.mean(df['price']), c='red', ls='--', label='mean')
plt.axvline(x=np.percentile(df['price'],25),c='green', ls='--', label = '25th percentile:Q1')
plt.axvline(x=np.percentile(df['price'],75),c='orange', ls='--',label = '75th percentile:Q3' )
plt.legend()


# In[31]:


#Scatter plot
plt.figure(figsize=(30,8))
df.plot(kind='scatter',x='date',y='price')
plt.show()


# In[32]:


#Observation : From above visualization we can see that there is variation in the gold price .


# In[33]:


#Let visualizing the sum of all sales each year. We can do that using group of “Price” and “Date” and group by “Year”.


# In[34]:


df=df
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday
df['weekday'] = np.where(df.weekday == 0, 7, df.weekday)
df_year = df[['price','year']].groupby(by='year').sum().reset_index()

df_year


# In[35]:


sns.catplot(x='year',y='price',data=df_year,kind='bar',aspect=2)


# In[36]:


#Monthly & Yearly Gold Price (TREND & SEASONALITY)
# Draw Plot
fig, axes = plt.subplots(1, 2, figsize=(20,7), dpi= 80)
sns.boxplot(x='year', y='price', data=df, ax=axes[0])
sns.boxplot(x='month', y='price', data=df.loc[~df.year.isin([2016, 2021]), :])

# Set Title
axes[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize=18); 
axes[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize=18)
plt.show()


# In[37]:


#Weekly Gold Price
plot = sns.boxplot(x='weekday', y='price', data=df)
plot.set(title='Weekly Gold Price')


# In[38]:


#Correlation Plot - EDA

#Finally, to find the correlation among the variables, we can make use of the correlation function.
#This will give you a fair idea of the correlation strength between different variables.


# In[39]:


df.corr()


# In[40]:


#This is the correlation matrix with the range from +1 to -1 where +1 is highly and positively correlated and -1 will be highly negatively correlated


# In[41]:


#Correlation plot

corr_matrix = df.corr()

plt.figure(figsize=(8,8))
sns.heatmap(data= corr_matrix,annot=True,vmin=0)
plt.show


# In[42]:


#Feature Enggineering


# In[43]:


# Making thr date as DateTime index for the DataFrame

df1 = df.copy()
df1.set_index('date',inplace=True)
df1.index.year


# In[44]:


visual = df1.copy()
visual.reset_index(inplace=True)
visual['date'] = pd.to_datetime(visual['date'])
visual['year'] = visual['date'].dt.year
visual['month'] = visual['date'].dt.month
visual['week'] = visual['date'].dt.isocalendar().week
visual['quarter'] = visual['date'].dt.quarter
visual['day_of_week'] = visual['date'].dt.day_name()
visual.drop('date', axis =1 , inplace= True)
visual.head(10)


# In[45]:


visual.year.unique


# In[46]:


#Average Price of Gold for each year


# In[47]:


df_2016 = visual[visual['year']==2016][['month','price']]
df_2016 = df_2016.groupby('month').agg({"price" : "mean"}).reset_index().rename(columns={'price':'2016'})
df_2017 = visual[visual['year']==2017][['month','price']]
df_2017 = df_2017.groupby('month').agg({"price" : "mean"}).reset_index().rename(columns={'price':'2017'})
df_2018 = visual[visual['year']==2018][['month','price']]
df_2018 = df_2018.groupby('month').agg({"price" : "mean"}).reset_index().rename(columns={'price':'2018'})
df_2019 = visual[visual['year']==2019][['month','price']]
df_2019 = df_2019.groupby('month').agg({"price" : "mean"}).reset_index().rename(columns={'price':'2019'})
df_2020 = visual[visual['year']==2020][['month','price']]
df_2020 = df_2020.groupby('month').agg({"price" : "mean"}).reset_index().rename(columns={'price':'2020'})
df_2021 = visual[visual['year']==2021][['month','price']]
df_2021 = df_2021.groupby('month').agg({"price" : "mean"}).reset_index().rename(columns={'price':'2021'})

df_year = df_2016.merge(df_2017,on='month').merge(df_2018,on='month').merge(df_2019,on='month').merge(df_2020,on='month').merge(df_2021,on='month')


# In[48]:


import plotly.graph_objects as go

# top levels
top_labels = ['2016', '2017', '2018', '2019', '2020','2021']

colors = ['rgb(6, 19, 14)', 'rgb(18, 58, 43)',
          'rgb(31, 97, 71)', 'rgb(43, 136, 100)',
          'rgb(55, 174, 129)','rgb(81, 200, 154)',
          'rgb(119, 212, 176)','rgb(158, 224, 199)']

# X axis value 
df_year = df_year[['2016', '2017', '2018', '2019', '2020','2021']].replace(np.nan,0)
x_data = df_year.values

# y axis value (Month)
df_2016['month'] =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
y_data = df_2016['month'].tolist()

fig = go.Figure()
for i in range(0, len(x_data[0])):
    for xd, yd in zip(x_data, y_data):
        fig.add_trace(go.Bar(
            x=[xd[i]], y=[yd],
            orientation='h',
            marker=dict(
                color=colors[i],
                line=dict(color='rgb(248, 248, 249)', width=1)
            )
        ))


# In[49]:


fig.update_layout(title='Avg Price for each Year',
    xaxis=dict(showgrid=False, 
               zeroline=False, domain=[0.15, 1]),
    yaxis=dict(showgrid=False, showline=False,
               showticklabels=False, zeroline=False),
    barmode='stack', 
    template="plotly_white",
    margin=dict(l=0, r=50, t=100, b=10),
    showlegend=False, 
)

annotations = []
for yd, xd in zip(y_data, x_data):
    # labeling the y-axis
    annotations.append(dict(xref='paper', yref='y',
                            x=0.14, y=yd,
                            xanchor='right',
                            text=str(yd),
                            font=dict(family='Arial', size=14,
                                      color='rgb(67, 67, 67)'),
                            showarrow=False, align='right'))
    # labeling the first Likert scale (on the top)
    if yd == y_data[-1]:
        annotations.append(dict(xref='x', yref='paper',
                                x=xd[0] / 2, y=1.1,
                                text=top_labels[0],
                                font=dict(family='Arial', size=14,
                                          color='rgb(67, 67, 67)'),
                          showarrow=False))
    space = xd[0]  
    for i in range(1, len(xd)):
            # labeling the Likert scale
            if yd == y_data[-1]:
                annotations.append(dict(xref='x', yref='paper',
                                        x=space + (xd[i]/2), y=1.1,
                                        text=top_labels[i],
                                        font=dict(family='Arial', size=14,
                                                  color='rgb(67, 67, 67)'),
                                        showarrow=False))
            space += xd[i]
fig.update_layout(
    annotations=annotations)
fig.show()


# In[50]:


#Observation:

#Highest prices of gold are in Auguest to December and the decreased in January to March
#Price are increasing gradually from 2019 to 2021
#Charts for visualise the Month,Quarter and week wise Price of Gold


# In[51]:


# data

import calendar


df_m_sa = visual.groupby('month').agg({"price" : "mean"}).reset_index()
df_m_sa['price'] = round(df_m_sa['price'],2)
df_m_sa['month_text'] = df_m_sa['month'].apply(lambda x: calendar.month_abbr[x])
df_m_sa['text'] = df_m_sa['month_text'] + ' - ' + df_m_sa['price'].astype(str) 

df_w_sa = visual.groupby('week').agg({"price" : "mean"}).reset_index() 
df_q_sa = visual.groupby('quarter').agg({"price" : "mean"}).reset_index() 
# chart color
df_m_sa['color'] = '#496595'
df_m_sa['color'][:-1] = '#c6ccd8'
df_w_sa['color'] = '#c6ccd8'


# In[52]:


from plotly.subplots import make_subplots


fig = make_subplots(rows=2, cols=2, vertical_spacing=0.1,
                    row_heights=[0.7, 0.3], 
                    specs=[[{"type": "bar"}, {"type": "pie"}],
                           [{"colspan": 2}, None]],
                    column_widths=[0.7, 0.3],
                    subplot_titles=("Month wise Avg Price Analysis", "Quarter wise Avg Price Analysis", 
                                    "Week wise Avg Price Analysis"))

fig.add_trace(go.Bar(x=df_m_sa['price'], y=df_m_sa['month'], marker=dict(color= df_m_sa['color']),
                     text=df_m_sa['text'],textposition='auto',
                     name='Month', orientation='h'), 
                     row=1, col=1)
fig.add_trace(go.Pie(values=df_q_sa['price'], labels=df_q_sa['quarter'], name='Quarter',
                     marker=dict(colors=['#334668','#496595','#6D83AA','#91A2BF','#C8D0DF']), hole=0.7,
                     hoverinfo='label+percent+value', textinfo='label+percent'), 
                     row=1, col=2)
fig.add_trace(go.Scatter(x=df_w_sa['week'], y=df_w_sa['price'], mode='lines+markers', fill='tozeroy', fillcolor='#c6ccd8',
                     marker=dict(color= '#496595'), name='Week'), 
                     row=2, col=1)

# styling
fig.update_yaxes(visible=False, row=1, col=1)
fig.update_xaxes(visible=False, row=1, col=1)
fig.update_xaxes(tickmode = 'array', tickvals=df_w_sa.week, ticktext=[i for i in range(1,53)], 
                 row=2, col=1)
fig.update_yaxes(visible=False, row=2, col=1)
fig.update_layout(height=750, bargap=0.15,
                  margin=dict(b=0,r=20,l=20), 
                  title_text="Average Price Analysis",
                  template="plotly_white",
                  title_font=dict(size=25, color='#8a8d93', family="Lato, sans-serif"),
                  font=dict(color='#8a8d93'),
                  hoverlabel=dict(bgcolor="#f2f2f2", font_size=13, font_family="Lato, sans-serif"),
                  showlegend=False)
fig.show()


# In[53]:


#Observation:
#As we saw in the above chart there is an upward trend in price of Gold over the time. Although there are ups and downs at every point in time, generally we can observe that the trend increases. Also we can notice how the ups and downs seem to be a bit regular, it means we might be observing a seasonal pattern here too. Let’s take a closer look by observing some year’s data:
#Highest price average price is on Tuesday.
#Auguest Month has the Highest price.


# In[ ]:





# In[54]:


#4) Model Building


# In[55]:


#Decomposition of Time series
from pylab import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose


# In[56]:


plt.rcParams.update({'figure.figsize':(18,8), 'figure.dpi':75})
result = seasonal_decompose(df_eda, model='additive', period=120)
result.plot()
plt.show()


# In[57]:


#Trend - Slow moving changes in a time series, Responisble for making series gradually increase or decrease over time.

#Seasonality - Seasonal Paterns in the series. The cycles occur repeatedly over a fixed period of time.

#Residuals - The behaviour of the time series that cannot be explained by the trend and seasonality components. Also called random errors/white noise.


# In[58]:


####Plotting Rolling Statistics


# In[59]:


rolmean = df['price'].rolling(12).mean()
rolstd = df['price'].rolling(12).std()

#Plot rolling statistics:

orig = plt.plot(df['price'], color='blue',label='Original')
mean = plt.plot(rolmean, color='red', label='Rolling Mean')
std = plt.plot(rolstd, color='black', label = 'Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)


# In[60]:


#We observe that the rolling mean and Standard deviation are not constant with respect to time (increasing trend)The time series is hence not stationary
#Testing for Stationarity

#Time Series is Stationary if we have constant mean, constant variance and No Trend and No Seasonality

#But in our data set we can see uprising trend and also seasonality is present, So we can say that our data is Non-Stationary.


# In[61]:


#ADF(Augmented Dickey-Fuller) Test


# In[62]:


from statsmodels.tsa.stattools import adfuller


# In[63]:


test_result=adfuller(df['price'])


# In[64]:


#Ho: It is non stationary
#H1: It is stationary

def adfuller_test(price):
    result=adfuller(price)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print(" reject the null hypothesis.Data is stationasry")
    else:
        print(" accept null hypothesis. Data is Non-Stationary ")


# In[65]:


adfuller_test(df['price'])


# In[66]:


df['price First Difference'] = df['price'] - df['price'].shift(1)


# In[67]:


df['Seasonal First Difference']=df['price']-df['price'].shift(30)


# In[68]:


## Again test dickey fuller test
adfuller_test(df['Seasonal First Difference'].dropna())


# In[69]:


df['Seasonal First Difference'].plot()


# In[70]:


## Again test dickey fuller test for first differencing
adfuller_test(df['price First Difference'].dropna())


# In[71]:


#Our Data is now stationary


# In[72]:


###ACF and PACF plots


# In[73]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


# In[74]:


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = plot_acf(df['price First Difference'].iloc[1:],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = plot_pacf(df['price First Difference'].iloc[1:],lags=40,ax=ax2)


# In[75]:


###Spliting Data


# In[76]:


df = pd.date_range(start='1/1/2016', end='21/12/2021', freq='M')
df


# In[77]:


df = pd.read_csv("Gold_data.csv")
df


# In[78]:


df['date'] = pd.to_datetime(df['date'])
df


# In[79]:


df = df.set_index('date')


# In[80]:


train    =   df[df.index.year <= 2020] 
test     =   df[df.index.year > 2020]


# In[81]:


print(train.shape)
print(test.shape)


# In[82]:


plt.plot(train)
plt.plot(test)
plt.show()


# In[83]:


#ARIMA MODEL


# In[84]:


get_ipython().system('pip install pmdarima')


# In[85]:


# Figure out order for ARIMA Model
from pmdarima import auto_arima


# In[86]:


stepwise_fit = auto_arima(train, trace = True, suppress_warnings=True, seasonal=False)
stepwise_fit.summary()


# In[87]:


from statsmodels.tsa.arima.model import ARIMA
model_arima = ARIMA(train['price'],order = (4,1,2))
result = model_arima.fit()
result.summary()


# In[88]:


test_pred = pd.DataFrame(result.predict(len(train),len(train)+354,type='levels'))
test_pred.index = test.index
test_pred


# In[89]:


start = len(train)
end=len(train)+len(test)-1
test_pred = pd.DataFrame(result.predict(start = start, end=end, type='levels'))
test_pred_index = test.index
test_pred
test_pred.index = df.index[start:end+1]   # To print ouput in date format
print(test_pred)


# In[90]:


plt.figure(figsize=(12,5), dpi=80)
plt.plot(train, label = 'Train')
plt.plot(test, label='Test')
plt.plot(test_pred, label='Prediction')
plt.title('Actuals vs Prediction', size=30)
plt.legend(loc='upper left', fontsize=10)
plt.show()


# In[91]:


from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error,mean_absolute_error
from math import sqrt


# In[92]:


mse = mean_squared_error(test_pred['predicted_mean'],test['price'])
print(f'Mean Squared Error (MSE) = ',mse)
rmse = np.round(np.sqrt(mse),2)
print(f'Root Mean Squared Error (RMSE) = ',rmse)
mae = mean_absolute_error(test_pred.predicted_mean,test.price)
print(f'Mean Absolute Error (MAE)  = ', mae)
mape = mean_absolute_percentage_error(test_pred.predicted_mean,test.price)
print(f'Mean Absolute Percentage Error (MAPE)  = ', mape)


# In[93]:


#Forecast for the 30 Days


# In[94]:


forecast = result.predict(len(df), len(df)+31, type = 'levels')
forecast
index_future_dates = pd.date_range(start='2021-12-21', end = '2022-01-21')
forecast.index=index_future_dates
print(forecast)


# In[95]:


plt.figure(figsize=(12,5), dpi=80)
plt.plot(train, label = 'Train')
plt.plot(test, label='Test')
plt.plot(forecast, label='Prediction')
plt.title('Actuals vs Prediction')
plt.legend(loc='upper left', fontsize=10)
plt.show()


# In[96]:


forecast.plot(figsize=(12,5), legend=True)


# In[97]:


#SARIMA


# In[98]:


import itertools
p = range(0, 3)
d = range(1,2)
q = range(0, 3)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 22) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


# In[99]:


model_SA = sm.tsa.SARIMAX(train['price'], order=(0,1,2), seasonal_order=(1,1,1,22))
result_SA = model_SA.fit()
result_SA.summary()


# In[100]:


test_pred_SA = pd.DataFrame(result_SA.predict(len(train),len(train)+354,type='levels'))
test_pred_SA.index = test.index
test_pred_SA


# In[101]:


plt.figure(figsize=(12,5), dpi=80)
plt.plot(train['price'], label = 'Train')
plt.plot(test['price'], label='Test')
plt.plot(test_pred_SA, label='Prediction')
plt.title('Actuals vs Prediction')
plt.legend(loc='upper left', fontsize=10)
plt.show()


# In[102]:


mse = mean_squared_error(test_pred_SA['predicted_mean'],test['price'])
print(f'Mean Squared Error (MSE) = ',mse)
rmse = np.round(np.sqrt(mse),2)
print(f'Root Mean Squared Error (RMSE) = ',rmse)
mae = mean_absolute_error(test_pred_SA.predicted_mean,test.price)
print(f'Mean Absolute Error (MAE)  = ', mae)
mape = mean_absolute_percentage_error(test_pred_SA.predicted_mean,test.price)
print(f'Mean Absolute Percentage Error (MAPE)  = ', mape)


# In[103]:


#Forecast for next 30 Days


# In[104]:


forecast_SA = result_SA.predict(len(df), len(df)+31, type = 'levels')
forecast_SA
index_future_dates = pd.date_range(start='2021-12-21', end = '2022-01-21')
forecast_SA.index=index_future_dates
print(forecast_SA)


# In[105]:


plt.figure(figsize=(12,5), dpi=80)
plt.plot(train['price'], label = 'Train')
plt.plot(test['price'], label='Test')
plt.plot(forecast_SA, label='Prediction')
plt.title('Actuals vs Prediction')
plt.legend(loc='upper left', fontsize=10)
plt.show()


# In[106]:


#Holt Method


# In[107]:


# single exponential smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
# double and triple exponential smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# In[108]:


'''Before starting with the models, we shall first define the weight coefficient Alpha and the Time Period. We also set the DateTime frequency to a monthly level.
#### Set the value of Alpha and define m (Time Period)'''
m = 22
alpha = 1/(2*m)


# In[109]:


# Single/Simple Exponential Method
df['ses_model'] = SimpleExpSmoothing(train["price"]).fit(smoothing_level=alpha, optimized=False,use_brute=True).fittedvalues
df[['ses_model','price']].plot(title='Holt Winters Single Exponential Smoothing', legend=True)
# df2['ses_model']


# In[110]:


#Type Markdown and LaTeX:  𝛼2


# In[111]:


df['ADD'] = ExponentialSmoothing(train['price'],trend='add').fit().fittedvalues
df['MUL'] = ExponentialSmoothing(train['price'],trend='mul').fit().fittedvalues
df[['price','ADD','MUL']].plot(title='Holt Winters Double Exponential Smoothing:Additive & Multiplicative Trend')


# In[112]:


#Type Markdown and LaTeX:  𝛼2
 
#Type Markdown and LaTeX: 𝛼2


# In[113]:


# Fit the model tend='mul', season='mul'
fitted_model = ExponentialSmoothing(train,trend='mul',seasonal='mul',seasonal_periods=11).fit()
test_predictions = fitted_model.forecast(355)
test_predictions.index = df['price'].index[start:end+1]   # To print ouput in date format
print(test_predictions)


# In[114]:


plt.figure(figsize=(12,5), dpi=80)
plt.plot(train, label = 'Train')
plt.plot(test, label='Test')
plt.plot(test_predictions, label='Prediction')
plt.title('Actuals vs Prediction')
plt.legend(loc='upper left', fontsize=10)
plt.show()


# In[115]:


mse = mean_squared_error(test,test_predictions)
print(f'Mean Squared Error (MSE) = ',mse)
rmse = np.round(np.sqrt(mse),2)
print(f'Root Mean Squared Error (RMSE) = ',rmse)
mae = mean_absolute_error(test,test_predictions)
print(f'Mean Absolute Error (MAE)  = ', mae)
mape = mean_absolute_percentage_error(test,test_predictions)
print(f'Mean Absolute Percentage Error (MAPE)  = ', mape)


# In[116]:


# Fit the model tend='add', season='mul'
fitted_model = ExponentialSmoothing(train,trend='add',seasonal='mul',seasonal_periods=12).fit()
test_predictions = fitted_model.forecast(355)
test_predictions.index = df['price'].index[start:end+1]  

 # To print ouput in date format
print(test_predictions)


# In[117]:


plt.figure(figsize=(12,5), dpi=80)
plt.plot(train, label = 'Train')
plt.plot(test, label='Test')
plt.plot(test_predictions, label='Prediction')
plt.title('Actuals vs Prediction')
plt.legend(loc='upper left', fontsize=10)
plt.show()


# In[118]:


mse = mean_squared_error(test,test_predictions)
print(f'Mean Squared Error (MSE) = ',mse)
rmse = np.round(np.sqrt(mse),2)
print(f'Root Mean Squared Error (RMSE) = ',rmse)
mae = mean_absolute_error(test,test_predictions)
print(f'Mean Absolute Error (MAE)  = ', mae)
mape = mean_absolute_percentage_error(test,test_predictions)
print(f'Mean Absolute Percentage Error (MAPE)  = ', mape)


# In[119]:


# Fit the model tend='mul', season='add'
fitted_model = ExponentialSmoothing(train,trend='mul',seasonal='add',seasonal_periods=12).fit()
test_predictions = fitted_model.forecast(355)
test_predictions.index = df['price'].index[start:end+1]   # To print ouput in date format
print(test_predictions)


# In[120]:


plt.figure(figsize=(12,5), dpi=80)
plt.plot(train, label = 'Train')
plt.plot(test, label='Test')
plt.plot(test_predictions, label='Prediction')
plt.title('Actuals vs Prediction')
plt.legend(loc='upper left', fontsize=10)
plt.show()


# In[121]:


mse = mean_squared_error(test,test_predictions)
print(f'Mean Squared Error (MSE) = ',mse)
rmse = np.round(np.sqrt(mse),2)
print(f'Root Mean Squared Error (RMSE) = ',rmse)
mae = mean_absolute_error(test,test_predictions)
print(f'Mean Absolute Error (MAE)  = ', mae)
mape = mean_absolute_percentage_error(test,test_predictions)
print(f'Mean Absolute Percentage Error (MAPE)  = ', mape)


# In[122]:


# Fit the model tend='add', season='add'
fitted_model = ExponentialSmoothing(train,trend='add',seasonal='add',seasonal_periods=10).fit()
test_predictions = fitted_model.forecast(355)
test_predictions.index = df['price'].index[start:end+1]   # To print ouput in date format
print(test_predictions)


# In[123]:


plt.figure(figsize=(12,5), dpi=80)
plt.plot(train, label = 'Train')
plt.plot(test, label='Test')
plt.plot(test_predictions, label='Prediction')
plt.title('Actuals vs Prediction')
plt.legend(loc='upper left', fontsize=10)
plt.show()


# In[124]:


mse = mean_squared_error(test,test_predictions)
print(f'Mean Squared Error (MSE) = ',mse)
rmse = np.round(np.sqrt(mse),2)
print(f'Root Mean Squared Error (RMSE) = ',rmse)
mae = mean_absolute_error(test,test_predictions)
print(f'Mean Absolute Error (MAE)  = ', mae)
mape = mean_absolute_percentage_error(test,test_predictions)
print(f'Mean Absolute Percentage Error (MAPE)  = ', mape)


# In[125]:


#EMA


# In[126]:


df['EMA'] = train['price'].ewm(span=22).mean()


# In[127]:


df['EMA']


# In[128]:


df['EMA_t'] = test['price'].ewm(span=22).mean()


# In[129]:


df['EMA_t']


# In[130]:


plt.figure(figsize=(12,5), dpi=80)
plt.grid(True)
plt.plot(train['price'],label='Train Data Closing Price')
plt.plot(test['price'],label='Test Data Closing Price')
plt.plot(df['EMA_t'],label='EMA')
plt.title('Actuals vs Prediction')
plt.legend(loc='upper left', fontsize=10)
plt.legend(loc=2)


# In[131]:


df['EMA_t'].dropna()


# In[132]:


mse = mean_squared_error(test['price'],df['EMA_t'].dropna())
print(f'Mean Squared Error (MSE) = ',mse)
rmse = np.round(np.sqrt(mse),2)
print(f'Root Mean Squared Error (RMSE) = ',rmse)
mae = mean_absolute_error(test['price'],df['EMA_t'].dropna())
print(f'Mean Absolute Error (MAE)  = ', mae)
mape = mean_absolute_percentage_error(test['price'],df['EMA_t'].dropna())
print(f'Mean Absolute Percentage Error (MAPE)  = ', mape)


# In[ ]:




