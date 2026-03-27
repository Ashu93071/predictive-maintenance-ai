import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Load dataset:
columns=['unit_id','cycle']+[f'op_setting{i}' for i in range(1,4)]+[f'sensor_{i}' for i in range(1,22)]
# print(len(columns))

train=pd.read_csv(r"C:\Users\rushi\OneDrive\Desktop\DATA SCIENCE\Data sets\Nasa c-map turbo fan degradation\train_FD001.txt",sep=' ')
# print(train.info()) #give complete info about df.

#clean dataset:
train=train.dropna(axis=1) #removes all column which consists Null values.
# print(train.shape)
train.columns=columns
# print(train.head(10))

#Create RUL(remaining useful life):
max_cycle=train.groupby('unit_id')['cycle'].max().reset_index()
max_cycle.columns=['unit_id','max_cycle']
# print(max_cycle.shape)
train=train.merge(max_cycle,on='unit_id') #Right joint the max_cycle df in train df where unit_id are same/matching.

#Now we need the target variable i.e RUL(remaining useable life). max_cycle-cycle.
train['RUL']=train['max_cycle']-train['cycle']
#now we have target values for each row group by unit_ids.
# print(train.columns)

#Drop unneccesary columns
train.drop(['max_cycle'],axis=1,inplace=True)

#remove useless sensors: beacause some sensors are not changing(constant)
print(train.nunique())
# Drop columns where variation is very low: droping columns where unique values are <2.
low_var_cols=[col for col in train.columns if train[col].nunique()<2]
train.drop(columns=low_var_cols,inplace=True)
# print(train.nunique())

#remove weak features: features which has less corelation with Target.
corr=train.corr()['RUL'].sort_values(ascending=False) #correlation coefficient of data with our 'RUL' features and sorting in ascending order.
weak_features=corr[abs(corr)<0.5].index # abs give distance from 0 which turns - to + and + stay +, index give the index name where corr <0.5
print('Features which are less correlation with RUL',weak_features)
new_df=train.drop(columns=weak_features.difference(['unit_id']))
print('New_Df:\n',new_df.head())

#Visulize data.
import seaborn as sns
sns.heatmap(new_df.corr(),annot=True)
new_rul=new_df['RUL'].clip(upper=125) #clips changes all the RUL values to 125 which are greater than 125.
plt.scatter(new_df['cycle'],new_rul)
plt.show()
new_df['RUL']=new_rul
# print('RUL max:',new_df['RUL'].max())
# print('RUL min:',new_df['RUL'].min())
# print('cycle max:',new_df['cycle'].max())
# print('cycle min:',new_df['cycle'].min())

#The data of sensors are of different scale. we must make then scale in range(0,1).
#Scaling.
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
feature_cols=new_df.columns.difference(['unit_id','RUL','cycle']) #selecting columns where to apply normalization.By using difference, it will give only feature which are different than provide feature.
# print(feature_cols)
new_df[feature_cols]=scaler.fit_transform(new_df[feature_cols])

# Calculate a 10-cycle moving average for your strongest sensors
for col in feature_cols: #we are taking the avg of values of sensors which reduce the Noise if present.
    new_df[f'{col}_avg'] = new_df.groupby('unit_id')[col].transform(lambda x: x.rolling(window=10).mean()) #Here, we applying function on each value in each column for getting the avg value. Rolling(window=10) is looking for current(1)+previous(9) values and then getting mean of our actual value. If the previous values are less than 9 it will gives NaN values for current value.
    new_df=new_df.drop(columns=col)

# print(new_df.info())
new_df.dropna(inplace=True)

#Train test splits: spliting by machine not randomly
train_units=new_df['unit_id'].unique()
# print(len(train_units))
train_ids=train_units[:80] #split data with 80:20 for train,test
test_ids=train_units[80:]

train_data=new_df[new_df['unit_id'].isin(train_ids)] #gets data of only rows where unique unit_ids are present.
test_data=new_df[new_df['unit_id'].isin(test_ids)]

#Final data
print('train data columns:',train_data.columns)
print('test data columns:',test_data.columns)

X_train=train_data.drop(['RUL','unit_id','cycle'],axis=1) 
y_train=train_data['RUL']
# print(X_train.columns)

X_test=test_data.drop(['RUL','unit_id','cycle'],axis=1)
y_test=test_data['RUL']

#visualize
plt.figure(figsize=(8,6))
index=1
for i in X_train:
    plt.subplot(4,6,index)
    index+=1
    plt.scatter(X_train[i],y_train,c=y_train)
    plt.xlabel(f'{X_train[i]}',fontsize=5)
    plt.ylabel(f'{y_train}',fontsize=5)
    # plt.title(f'{X_train[i]} vs {y_train}',fontsize=3)
plt.tight_layout(pad=0.3)
plt.show()

#Model building:
#using random forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,root_mean_squared_error
rf=RandomForestRegressor()#(n_estimators=100,max_depth=25,min_samples_leaf=10,random_state=42,n_jobs=-1) #max_depth Prevents the model from memorizing noise, min_samples_leaf Ensures each "prediction" is based on at least 5 samples, n_jobs=-1 Uses all your CPU cores for speed
rf.fit(X_train,y_train)
y_pred_rf=rf.predict(X_test)
print('rf score:',rf.score(X_train,y_train))

#using linear regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)
y_pred_lr=lr.predict(X_test)
print('lr score:',lr.score(X_train,y_train))

#Evaluation
rf_mae=mean_absolute_error(y_test,y_pred_rf)
rf_rmse=root_mean_squared_error(y_test,y_pred_rf)
lr_mae=mean_absolute_error(y_test,y_pred_lr)
lr_rmse=root_mean_squared_error(y_test,y_pred_lr)

print('Random forest Mean Absoulute Error:',rf_mae,'\nRandom forest Root mean square error:',rf_rmse)
print('Linear Regression Mean Absoulute Error:',lr_mae,'\nLinear regression Root mean square error:',lr_rmse)


# Plot actual vs predicted: for Linear Regression
plt.scatter(y_pred_lr,y=np.arange(len(y_pred_lr)),color='red',label='Predicted')
plt.scatter(y_test,y=np.arange(len(y_pred_lr)),color='green',label='Actual')
plt.title('Linear Regression: Actual vs Predicted')
plt.legend()
plt.show()


# Plot actual vs predicted: for Random Forest
plt.scatter(y_pred_rf,y=np.arange(len(y_pred_rf)),color='red',label='Predicted')
plt.scatter(y_test,y=np.arange(len(y_pred_rf)),color='green',label='Actual')
plt.title('Random forest: Actual vs Predicted')
plt.legend()
plt.show()

# #Check features importance
importance=pd.Series(rf.feature_importances_,index=X_train.columns)
importance.sort_values().plot(kind='barh')
plt.title("Which Sensors Matter Most?")
plt.show()

# Calculate error for "Dangerous" low RUL vs "Safe" high RUL
test_results=X_test.copy()
test_results['Actual']=y_test
test_results['Predicted']=y_pred_rf
test_results['Error']=abs(test_results['Actual']-test_results['Predicted'])
print('MAE for engines near failure (RUL <30):',test_results[test_results['Actual']<30 ]['Error'].mean())
print('MAE for engines are healthy (RUL >100):',test_results[test_results['Actual']>100 ]['Error'].mean())

#Save model:
import joblib
joblib.dump(rf,r'DataScience\Predictive Maintenance with ML + Time Series\saved_model\rf_model.pkl')
joblib.dump(scaler,r'DataScience\Predictive Maintenance with ML + Time Series\saved_model\scaler.pkl')
joblib.dump(feature_cols,r'DataScience\Predictive Maintenance with ML + Time Series\saved_model\features_names.pkl')

print(y_test.iloc[0]) #RUL value 125
sample_x_test=scaler.inverse_transform([X_train.iloc[0]])
print(sample_x_test) #y_train values[ 47.24     522.12    2388.043      8.40984  391.8      642.24738.972     23.39557 1586.917   1400.844    554.114   2388.048]

def get_train_test_values(min_rul=0,max_rul=125):
    test_feat_ind={'index':[],'values':[]}
    for i in range(0,len(y_test)):
        value=y_test.iloc[i]
        if value in range(min_rul,max_rul):
            test_feat_ind['index'].append(i)
            test_feat_ind['values'].append(float(y_test.iloc[i]))

    index=test_feat_ind['index'][0] #index of target value in range min and max RUL
    index_value=test_feat_ind['values'][0] #RUL of target index position

    X_test_values=X_test.iloc[index]
    scaled_X_test_value=scaler.inverse_transform([X_test_values])

    return {'RUL_value':index_value,'Features_value':scaled_X_test_value}

print(get_train_test_values(60,100))