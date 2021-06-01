import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings( "ignore")
%matplotlib inline


from sklearn.preprocessing import MinMaxScaler,OneHotEncoder,LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score,roc_auc_score
from xgboost import XGBClassifier


df = pd.read_csv('/train_s3TEQDk.csv')
tf = pd.read_csv('test_mSzZ8RL.csv')

print(df.info())
print(tf.info())

#Check Null Values in data

print("Number of null values in Credit_Product in training set are:",df["Credit_Product"].isna().sum(),
      "-- Proportion",round(df["Credit_Product"].isna().sum()/len(df),2))

print("Number of null values in Credit_Product in test set are:",tf["Credit_Product"].isna().sum(),
      "-- Proportion",round(tf["Credit_Product"].isna().sum()/len(tf),2))

#distribution plots

sns.kdeplot(data=df['Age'],shade=True)
plt.ylim(0,0.1)

sns.kdeplot(tf['Age'],shade=True)
plt.ylim(0,0.1)

plt.figure(figsize=(20,11))
sns.scatterplot(x=df["Region_Code"],y=df["Avg_Account_Balance"]/10**4,hue=df["Is_Lead"])
plt.xticks(rotation=45)

plt.figure(figsize=(20,11))
sns.countplot(x=df["Region_Code"],hue=df["Is_Lead"])
plt.xticks(rotation=45)

df.loc[df["Is_Lead"]==1]["Avg_Account_Balance"].mean()

df.loc[df["Is_Lead"]==0]["Avg_Account_Balance"].mean()

region_propor = pd.DataFrame((df.groupby(['Region_Code','Is_Lead']).size() / df.groupby('Region_Code').size()).unstack(level=[1]))
region_propor.round(2)

RP = region_propor[1].round(2)

#Region_Prob = RP.to_dict()
Region_Prob = {'RG250': 0.16,
 'RG251': 0.23,
 'RG252': 0.14,
 'RG253': 0.26,
 'RG254': 0.21,
 'RG255': 0.23,
 'RG256': 0.14,
 'RG257': 0.19,
 'RG258': 0.22,
 'RG259': 0.19,
 'RG260': 0.19,
 'RG261': 0.17,
 'RG262': 0.18,
 'RG263': 0.22,
 'RG264': 0.14,
 'RG265': 0.26,
 'RG266': 0.14,
 'RG267': 0.15,
 'RG268': 0.3,
 'RG269': 0.22,
 'RG270': 0.14,
 'RG271': 0.17,
 'RG272': 0.22,
 'RG273': 0.24,
 'RG274': 0.16,
 'RG275': 0.17,
 'RG276': 0.28,
 'RG277': 0.23,
 'RG278': 0.22,
 'RG279': 0.23,
 'RG280': 0.24,
 'RG281': 0.22,
 'RG282': 0.19,
 'RG283': 0.3,
 'RG284': 0.3}


# distribution plot for average account balance
sns.kdeplot(x=df["Avg_Account_Balance"]/10**4,hue=df["Is_Lead"])
plt.ylim(-0.001,0.008)
plt.show()

# concatenating both data to perform scaling and OHC
df_all = pd.concat([df,tf],axis=0)

df_all["Region_Code"].unique()==df["Region_Code"].unique()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df["R_Codes"] = le.fit_transform(df["Region_Code"])
df["R_Codes"].dtype

df_all["R_Codes"] = le.fit_transform(df_all["Region_Code"])
df_all["R_Codes"].dtype

plt.figure(figsize=(20,10))
sns.scatterplot(x=df["R_Codes"],y = df["Avg_Account_Balance"]/10**3,hue = df["Is_Lead"])

plt.figure(figsize=(20,10))
sns.scatterplot(x=df["Occupation"],y = df["Avg_Account_Balance"]/10**3,hue = df["Is_Lead"])

regbal1 = pd.DataFrame(df.loc[df["Avg_Account_Balance"]<400000].groupby(["R_Codes","Is_Lead"]).size().unstack(level=[1]))
regbal2 = pd.DataFrame(df.loc[df["Avg_Account_Balance"]>400000].groupby(["R_Codes","Is_Lead"]).size().unstack(level=[1]))

axs[0]=regbal1.plot()
axs[1]=regbal2.plot()
plt.show()

(df_all.Avg_Account_Balance/10**3).median()

#categorizing continuous data.
bins = [20000,400000,1000000,5000000,10353000]
labels = ['low','medium','rich','uber']
df_all["balance_category"] = pd.cut(df.Avg_Account_Balance,bins=bins,labels=labels,right=True)

df_all["balance_category"].value_counts()

balance_propor = pd.DataFrame((df.groupby(['balance_category','Is_Lead']).size() / df.groupby(['Is_Lead']).size()).unstack(level=[1]))
balance_propor.round(2)


columns = df.columns.drop(['ID','Is_Lead'])

columns

# defining columns for preprocessing
num_cols = ["Age","Vintage","Reg_Prob"]
sca_num_cols = ["Age","Vintage"]
cat_cols = ["Gender","Occupation",'Channel_Code',"Bal_Cat","Credit_Product","Is_Active"]

corr = df_all.corr()
sns.heatmap(corr)

scaler = MinMaxScaler()

df_all[sca_num_cols]=scaler.fit_transform(df_all[sca_num_cols])


ohc = OneHotEncoder(handle_unknown="ignore",sparse=False)

OHC_df_all = pd.DataFrame(ohc.fit_transform(df_all[cat_cols]))
OHC_df_all.columns = ohc.get_feature_names(cat_cols)
OHC_df_all.index = df_all.index

catcols=OHC_df_all.columns
OHC_df_all.head()

df_final = pd.concat([df_all,OHC_df_all],axis=1)

#separating training and testing dataset.
test = df_final.loc[df_final["Is_Lead"].isna()]
test = test.drop("Is_Lead",axis=1)
test_fit = pd.concat([test[num_cols],test[catcols]],axis=1)

train = df_final.loc[~df_final["Is_Lead"].isna()]

y = train["Is_Lead"].astype("int64")
X = pd.concat([train[num_cols],train[catcols]],axis=1)

# splitting data for training and validation.
X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.2,random_state=49)

#**Models**
rfc1 = RandomForestClassifier(n_estimators=100,max_features='auto',max_depth=12,
                             criterion = 'gini',random_state=1)
xgbc = XGBClassifier()
rfc2 = RandomForestClassifier(n_estimators=100,max_features='log2',max_depth=12,
                             criterion = 'entropy',random_state=49)
lgc = LogisticRegression()
#ngb = ComplementNB()- discarded for low performance.
#svc = SVC() - discarded for execution cost

# voting classifier
estimators = [('rfc1',rfc1),('rfc2',rfc2),('svc',svc),('lr',lgc)]

#xgbc = XGBClassifier()

#"""rfc2 = RandomForestClassifier(n_estimators=100,max_features='log2',max_depth=9,
                           #  criterion = 'entropy',random_state=49)"""

#cv_rfc = GridSearchCV(estimator=rfc,param_grid=param_grid,cv=5)

#cv_rfc.fit(X_train,y_train)

#rfc2.fit(X_train,y_train)

#rfc1.fit(X_train,y_train)

#pred1 = rfc1.predict(X_valid)

#print("ROC AUC Score is",roc_auc_score(pred1,y_valid))

#pred = rfc2.predict(X_valid)

#print("ROC AUC Score is",roc_auc_score(pred,y_valid))

#svc.fit(X_train,y_train)

#pred_svc = svc.predict(X_valid)
#print("ROC AUC Score is",roc_auc_score(pred_svc,y_valid))

vc = VotingClassifier(estimators=estimators,voting='soft')

vc.fit(X_train,y_train)

pred_ens = vc.predict(X_valid)

print("ROC AUC Score is",roc_auc_score(pred_ens,y_valid))

#train[num_cols].plot(kind='box')
#plt.show()

predicted = vc.predict(test_fit)

#preparation for neural network
Xt = np.asarray(X_train)
yt = np.asarray(y_train)
Xv = np.asarray(X_valid)
yv = np.asarray(y_valid)
tt = np.asarray(test_fit)

import tensorflow
from tensorflow import keras

# model - NeuralNetwork
model = keras.Sequential([
     keras.layers.Dense(16, activation = 'relu', input_dim=Xt.shape[1]),
     keras.layers.Dense(64, activation = 'relu'),
     #keras.layers.Dropout(rate=0.15),
     keras.layers.Dense(128, activation = 'relu'),
     keras.layers.Dropout(rate=0.1),
     keras.layers.Dense(128, activation = 'relu'),
     keras.layers.Dropout(rate=0.15),
     keras.layers.Dense(128, activation = 'relu'),
     keras.layers.Dense(64,activation = 'relu'),
     keras.layers.Dropout(rate=0.15),
     keras.layers.Dense(16, activation = 'relu'),
     #keras.layers.Dropout(rate=0.1),
     keras.layers.Dense(1, activation = 'sigmoid')
 ])

#compiling model
 model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(lr=0.002), metrics=['accuracy'])
 model.summary()

callbacks = tensorflow.keras.callbacks.EarlyStopping(monitor='loss',min_delta = 0.001,patience=5)

history = model.fit(Xt, yt, epochs=100, verbose=2,callbacks=[callbacks])

df = pd.DataFrame(history.history)
df.plot()
plt.show()

model.evaluate(X_valid,y_valid)

pred_dl = model.predict(tt)

pred_dl.shape

print("ROC AUC Score is",roc_auc_score(pred_dl,predicted))

output = pd.DataFrame()
output['ID'] = tf['ID']
output['Is_Lead'] = pred_dl.round().astype(int)
output.head()


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-30T18:16:00.533596Z","iopub.status.idle":"2021-05-30T18:16:00.534157Z"}}
output.to_csv('submission.csv',index=False)
