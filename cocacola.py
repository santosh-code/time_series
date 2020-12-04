import pandas as pd
cocacola = pd.read_csv("C:/Users/USER/Desktop/TIME_SERIES/CocaCola_Sales_Rawdata.csv")

# Pre processing
import numpy as np

quarter=['Q1','Q2','Q3','Q4']
n=cocacola['Quarter'][0]
n[0:2]

cocacola['quarter']=0

for i in range(42):
    n=cocacola['Quarter'][i]
    cocacola['quarter'][i]=n[0:2]
    
dummy=pd.DataFrame(pd.get_dummies(cocacola['quarter']))



cocacola["t"] = np.arange(1,43)

cocacola["t_squared"] = cocacola["t"]*cocacola["t"]
cocacola["log_sales"] = np.log(cocacola["Sales"])
cocacola.columns

# Visualization - Time plot
coco=pd.concat((cocacola,dummy),axis=1)
coco.Sales.plot()

# Data Partition
Train = coco.iloc[0:38,:]
Test  =coco.iloc[38:42,:]

# to change the index value in pandas data frame 
# Test.set_index(np.arange(1,13))

####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Sales ~ t', data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales']) - np.array(pred_linear))**2))
rmse_linear

##################### Exponential ##############################

Exp = smf.ols('log_sales ~ t', data = Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales']) - np.array(np.exp(pred_Exp)))**2))
rmse_Exp

#################### Quadratic ###############################

Quad = smf.ols('Sales ~ t+t_squared', data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad

################### Additive seasonality ########################

additive= smf.ols('Sales~ Q1+Q2+Q3',data=Train).fit()
predadd=pd.Series(additive.predict(pd.DataFrame(Test[['Q1','Q2','Q3','Q4']])))
predadd
rmseadd=np.sqrt(np.mean((np.array(Test['Sales'])-np.array(predadd))**2))
rmseadd

################## Multiplicative Seasonality ##################

mulsea=smf.ols('log_sales~Q1+Q2+Q3+Q4',data=Train).fit()
predmul= pd.Series(mulsea.predict(pd.DataFrame(Test[['Q1','Q2','Q3','Q4']])))
rmsemul= np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(predmul)))**2))
rmsemul

################## Additive Seasonality Quadratic Trend ############################

addquad=smf.ols('Sales~t+t_squared+Q1+Q2+Q3+Q4',data=Train).fit()
predaddquad=pd.Series(addquad.predict(pd.DataFrame(Test[['t','t_squared','Q1','Q2','Q3','Q4']])))
rmseaddquad=np.sqrt(np.mean((np.array(Test['Sales'])-np.array(predaddquad))**2))
rmseaddquad

################## Multiplicative Seasonality Linear Trend  ###########

mullin= smf.ols('log_sales~t+Q1+Q2+Q3+Q4',data=Train).fit()
predmullin= pd.Series(mullin.predict(pd.DataFrame(Test[['t','Q1','Q2','Q3','Q4']])))
rmsemulin=np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(predmullin)))**2))
rmsemulin


data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmseadd","rmsemul","rmseaddquad","rmsemulin"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmseadd,rmsemul,rmseaddquad,rmsemulin])}
table_rmse=pd.DataFrame(data)
table_rmse