############## FORECASTING  AIRLINES DATA ####################


###### IMPORTING REQUIRED PACKAGES AND LOADING THE DATA #########

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # 
import statsmodels.graphics.tsaplots as tsa_plots
import statsmodels.tsa.statespace as tm_models
from datetime import datetime,time
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm


Airlines_Data = pd.read_csv("C:\\Users\\home\\Desktop\\Data Science Assignments\\Python_codes\\Forecasting\\Airlines+Data.csv")

Airlines_Data




Airlines_Data.index = pd.to_datetime(Airlines_Data.Month,format="%b-%y")

colnames = Airlines_Data.columns
colnames #Index(['Month', 'Passengers'], dtype='object')

Airlines_Data.Passengers.plot() # time series plot 
# Creating a Date column to store the actual Date format for the given Month column
Airlines_Data["Date"] = pd.to_datetime(Airlines_Data.Month,format="%b-%y")

# Extracting Day, weekday name, month name, year from the Date column using 
# Date functions from pandas 

Airlines_Data["month"] = Airlines_Data.Date.dt.strftime("%b") # month extraction
#Amtrak["Day"] = Amtrak.Date.dt.strftime("%d") # Day extraction
#Amtrak["wkday"] = Amtrak.Date.dt.strftime("%A") # weekday extraction
Airlines_Data["year"] =Airlines_Data.Date.dt.strftime("%Y") # year extraction



############ or ############

#month =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] 

#b = Airlines_Data["Month"][0]
#b[0:3]
#Airlines_Data['months']= 0

#for i in range(96):
#    b= Airlines_Data["Month"][i]
#    Airlines_Data['months'][i]= b[0:3]
    
month_dummies = pd.DataFrame(pd.get_dummies(Airlines_Data['month']))
Airlines_Data1 = pd.concat([Airlines_Data,month_dummies],axis = 1)

Airlines_Data1["t"] = np.arange(1,97)

Airlines_Data1["t_squared"] = Airlines_Data1["t"]*Airlines_Data1["t"]
Airlines_Data1.columns#Index(['Month', 'Passengers', 'Date', 'month', 'year', 'Apr', 'Aug', 'Dec',
       #'Feb', 'Jan', 'Jul', 'Jun', 'Mar', 'May', 'Nov', 'Oct', 'Sep', 't',
       #'t_squared'],
      dtype='object')

Airlines_Data1["log_passengers"] = np.log(Airlines_Data1["Passengers"])

Airlines_Data1.rename(columns={"Passengers ": 'Passengers'}, inplace=True)
Airlines_Data1.Passengers.plot()
Train = Airlines_Data1.head(84)
Test = Airlines_Data1.tail(12)













########################################################################
################### M O D E L     B U I L D I N G ######################
########################################################################

####################### L I N E A R ##########################


linear_model = smf.ols('Passengers~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear))**2))
rmse_linear#### 53.19923653480265#error

##################### Exponential ##############################

Exp = smf.ols('log_passengers~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp#####: 46.05736110315619

#################### Quadratic ###############################

Quad = smf.ols('Passengers~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_Quad))**2))
rmse_Quad ####: 48.051888979331615

################### Additive seasonality ########################

add_sea = smf.ols('Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea##### 132.81978481421814

################## Additive Seasonality Quadratic ############################

add_sea_Quad = smf.ols('Passengers~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad ######### 26.360817612081384

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea #### 140.06320204708618

##################Multiplicative Additive Seasonality ###########

Mul_Add_sea = smf.ols('log_passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea   #####  10.519172544323684


###### Therefore the Multiplicative Additive Seasonality have the least mean squared error


################## Testing the model #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse

#Out[339]: 
#               MODEL  RMSE_Values
#0        rmse_linear    53.199237
#1           rmse_Exp    46.057361
#2          rmse_Quad    48.051889
#3       rmse_add_sea   132.819785
#4  rmse_add_sea_quad    26.360818
#5      rmse_Mult_sea   140.063202
#6  rmse_Mult_add_sea    10.519173
# so rmse_add_sea has the least value among the models prepared so far 








# Accuracy = Test
np.mean(pred_Mult_add_sea==Test.log_passengers) #  0.0
 
Test["Forecasted_Passengers"]=pd.Series(pred_Mult_add_sea)


Airlines_Data1["Forecasted_Passengers"]=pd.Series(pred_Mult_add_sea)






# Accuracy = train 
pred_Mult_add_sea2 = pd.Series(Mul_Add_sea.predict(Train))
np.mean(pred_Mult_add_sea2 == Train.log_passengers)
Airlines_Data1["Forecasted_Passengers"]=pd.Series(pred_Mult_add_sea2)



















##################################################################################
####################### D A T A   D R I V E N    M O D E L #######################
##################################################################################




# Some EDA on Time series data 
# Heat map visualization 
heatmap_y_month = pd.pivot_table(data=Airlines_Data,values="Passengers",index="year",columns="month",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g")

# Boxplot for ever
sns.boxplot(x="month",y="Passengers",data=Airlines_Data)
sns.boxplot(x="year",y="Passengers",data=Airlines_Data)
#sns.factorplot("month","Passengers",data=Airlines_Data,kind="box")

# Line plot for Passengers based on year  and for each month
sns.lineplot(x="year",y="Passengers",hue="month",data=Airlines_Data)


# moving average for the time series to understand better about the trend character in Airlines_Data
Airlines_Data.Passengers.plot(label="org")
for i in range(2,18,6):
    Airlines_Data["Passengers"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=2)



# Time series decomposition plot 
decompose_ts_add = seasonal_decompose(Airlines_Data.Passengers,model="additive")
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(Airlines_Data.Passengers,model="multiplicative")
decompose_ts_mul.plot()

# ACF plots and PACF plots on Original data sets 
tsa_plots.plot_acf(Airlines_Data.Passengers,lags=10)
tsa_plots.plot_pacf(Airlines_Data.Passengers)

 
# splitting the data into Train and Test data and considering the last 12 months data as 
# Test data and left over data as train data 

Train1 = Airlines_Data.head(84)
Test1 = Airlines_Data.tail(12)
# to change the index value in pandas data frame 
# Test.set_index(np.arange(1,13),inplace=True)







# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org))*100/org
    return np.mean(temp)



# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train1["Passengers"]).fit()
pred_ses = ses_model.predict(start = Test1.index[0],end = Test1.index[-1])
MAPE(pred_ses,Test1.Passengers) #  14.235433039401634

# Holt method 
hw_model = Holt(Train1["Passengers"]).fit()
pred_hw = hw_model.predict(start = Test1.index[0],end = Test1.index[-1])
MAPE(pred_hw,Test1.Passengers) # 11.840943119376163



# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train1["Passengers"],seasonal="add",trend="add",seasonal_periods=12,damped=True).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test1.index[0],end = Test1.index[-1])
MAPE(pred_hwe_add_add,Test1.Passengers) # 1.6126576094462026



# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train1["Passengers"],seasonal="mul",trend="add",seasonal_periods=12).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test1.index[0],end = Test1.index[-1])
MAPE(pred_hwe_mul_add,Test1.Passengers) # 2.819737161363712



                


# Visualization of Forecasted values for Test data set using different methods 
plt.plot(Train.index, Train["Passengers"], label='Train',color="black")
plt.plot(Test.index, Test["Passengers"], label='Test',color="blue")
plt.plot(pred_ses.index, pred_ses, label='SimpleExponential',color="green")
plt.plot(pred_hw.index, pred_hw, label='Holts_winter',color="red")
plt.plot(pred_hwe_add_add.index,pred_hwe_add_add,label="HoltsWinterExponential_1",color="brown")
plt.plot(pred_hwe_mul_add.index,pred_hwe_mul_add,label="HoltsWinterExponential_2",color="yellow")
plt.legend(loc='best')















