#!/usr/bin/env python
# coding: utf-8

# In[83]:


import numpy as np
import pandas as pd


# In[84]:


df=pd.read_csv("Bengaluru_House_Data.csv")


# In[85]:


df


# In[86]:


df.head()


# In[87]:


df.shape


# In[88]:


df.info()


# In[89]:


df.isnull().sum()


# In[90]:


df.drop(columns=["area_type","availability","society","balcony"],inplace=True)


# In[91]:


df


# In[92]:


df.describe()


# In[93]:


df.info()


# In[94]:


df["location"].value_counts()


# In[95]:


df["location"]=df["location"].fillna("Sarjapur  Road")


# In[ ]:





# In[96]:


df["size"]=df["size"].fillna("2 BHK")


# In[97]:


df["bath"]=df["bath"].fillna(df["bath"].median())


# In[98]:


df.info()


# In[99]:


df["bhk"]=df["size"].str.split().str.get(0).astype(int)


# In[102]:


df[df["bhk"] > 20]


# In[103]:


df["total_sqft"].unique()


# In[104]:


def convertRange(x):
    
    temp = x.split("-")
    if len(temp)==2:
        return(float(temp[0])+float(temp[1]))/2
    try:
        return float(x)
    except:
        return None


# In[105]:


df["total_sqft"]=df["total_sqft"].apply(convertRange)


# In[106]:


df.head()


# In[107]:


df["price_per_sqft"]=df["price"]*100000/df["total_sqft"]


# In[108]:


df["price_per_sqft"]


# In[109]:


df.describe()


# In[110]:


df["location"] = df["location"].apply(lambda x: x.strip())
location_count=df["location"].value_counts()


# In[111]:


location_count_less_10=location_count[location_count<=10]


# In[112]:


location_count_less_10


# In[113]:


df["location"]=df["location"].apply(lambda x: "other" if x in location_count_less_10 else x )


# In[114]:


df["location"].value_counts()


# # outlier detection

# In[115]:


df.describe()


# In[117]:


(df["total_sqft"]/df["bhk"]).describe()


# In[119]:


df=df[df["total_sqft"]/df["bhk"]>=300]


# In[120]:


df.describe()


# In[121]:


df.shape


# In[122]:


df.price_per_sqft.describe()


# In[123]:


def remove_outliers_sqft(df):
    df_output = pd.DataFrame()
    for key,subdf in df.groupby("location"):
        m = np.mean(subdf.price_per_sqft)
        
        st = np.std(subdf.price_per_sqft)
        
        gen_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
        df_output= pd.concat([df_output,gen_df],ignore_index=True)
    return df_output
df=remove_outliers_sqft(df)
df.describe()
    


# In[128]:


def bhk_outlier_remove(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby("location"):
        bhk_stats={}
        for bhk, bhk_df in location_df.groupby("bhk"):
            bhk_stats[bhk] = {
                "mean":np.mean(bhk_df.price_per_sqft),
                "std":np.std(bhk_df.price_per_sqft),
                "count":bhk_df.shape[0]
            }
        for bhk,bhk_df in location_df.groupby("bhk"):
            stats=bhk_stats.get(bhk-1)
            if stats and stats["count"]>5:
                exclude_indices = np.append(exclude_indices,bhk_df[bhk_df.price_per_sqft<(stats["mean"])].index.values)
    return df.drop(exclude_indices,axis=0)


# In[129]:


df=bhk_outlier_remove(df)


# In[131]:


df.shape


# In[ ]:


df.drop(columns=["size","price_per_sqft"],inplace=True)


# In[136]:


df


# In[137]:


df.to_csv("cleaned_data.csv")


# In[138]:


X=df.drop(columns=["price"])
y=df["price"]


# In[140]:


X.head(2)


# In[141]:


y.head(2)


# In[151]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score


# In[152]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# In[153]:


X_train.shape


# In[154]:


X_test.shape


# In[155]:


# applying linear regression


# In[160]:


column_trans=make_column_transformer((OneHotEncoder(sparse=False),["location"]),remainder="passthrough")


# In[163]:


scaler=StandardScaler()


# In[164]:


lr=LinearRegression()


# In[165]:


pipe=make_pipeline(column_trans,scaler,lr)


# In[166]:


pipe.fit(X_train,y_train)


# In[167]:


y_pred_lr=pipe.predict(X_test)


# In[168]:


r2_score(y_test,y_pred_lr)


# In[169]:


# applying lasso regression


# In[170]:


lasso= Lasso()


# In[171]:


pipe=make_pipeline(column_trans,scaler,lasso)


# In[172]:


pipe.fit(X_train,y_train)


# In[175]:


y_pred_lasso=pipe.predict(X_test)


# In[176]:


r2_score(y_test,y_pred_lasso)


# In[177]:


# applying ridge regression


# In[178]:


ridge=Ridge()


# In[179]:


pipe=make_pipeline(column_trans,scaler,ridge)


# In[180]:


pipe.fit(X_train,y_train)


# In[181]:


y_pred_ridge=pipe.predict(X_test)


# In[182]:


r2_score(y_test,y_pred_ridge)


# In[183]:


print("no_regularization",r2_score(y_test,y_pred_lr))
print("Lasso",r2_score(y_test,y_pred_lasso))
print("Ridge",r2_score(y_test,y_pred_ridge))


# In[184]:


import pickle


# In[186]:


pickle.dump(pipe,open("RidgeModel.pkl","wb"))


# In[ ]:




