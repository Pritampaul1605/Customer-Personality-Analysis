import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st

st.title("Customers Personality Analysis")
market_data = pd.read_excel(r"C:\Users\Pritam Paul\Downloads\marketing_campaign1 (1).xlsx")
st.dataframe(market_data)

# Droping 'ID' Column :
market_data.drop(['ID'], axis = 1, inplace = True)

# Droping Duplicates :
market_data = market_data.drop_duplicates()

# Filling nan values with median values
median_income = market_data['Income'].median()

# Filling nan values with median values
market_data['Income'] = market_data['Income'].fillna(median_income)

# Droping unnecessry Columns :
market_data.drop(['Z_CostContact','Z_Revenue'], axis = 1, inplace = True)

# Creating new column : 'Age'
market_data['Year'] = market_data.Dt_Customer.dt.strftime('%Y')
market_data['Year'] = market_data['Year'].astype(int)
market_data['Age'] = market_data['Year'] - market_data['Year_Birth']

# Droping Columns :
market_data.drop(['Year_Birth','Dt_Customer','Year'], axis = 1, inplace = True)



# Visulization :

st.sidebar.subheader("Visulization of Categorical Column : ")
categorical_column = ['Education','Marital_Status']
chart_select = st.sidebar.selectbox(
                           label = 'Type of Charts : ',
                           options = ['Bar Plot', 'Pie Plot']
               )
column = st.sidebar.radio("Select any Column : ", options = categorical_column)

if st.sidebar.button('Generate Plot'):
    if chart_select == 'Bar Plot' :
        st.subheader("Bar Plot : ", column)
        plot = plt.figure(figsize = (12,7))
        pp = market_data[column].value_counts().plot(kind = 'bar')
        for i in pp.containers :
            pp.bar_label(i,)
        st.pyplot(plot)

    if chart_select == 'Pie Plot' :
        st.subheader("Pie Plot : ", column)
        plots = plt.figure(figsize = (4,4))
        market_data[column].value_counts().plot(kind = 'pie', autopct='%1.2f%%')
        st.pyplot(plots)


st.sidebar.subheader("Visulization of Numerical Column : ")
numerical_column = ['Income','MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds']
chart_select = st.sidebar.radio(
                           label = 'Type of Charts : ',
                           options = ['Box Plot', 'Kde Plot', 'Histogram']
               )
column = st.sidebar.selectbox("Select any Column : ", options = numerical_column)


if st.sidebar.button('Generate Plot', key = 'chech_1'):
    if chart_select == 'Box Plot' :
        st.subheader("Box Plot : ")
        plot = px.box(market_data, x = column)
        st.write(plot)

    if chart_select == 'Kde Plot' :
        st.subheader("Kde Plot : ")
        plot = plt.figure(figsize = (10,6))
        sns.kdeplot(market_data[column],color = 'red')
        st.pyplot(plot)
        
    if chart_select == 'Histogram' :
        st.subheader("Histogram : ")
        plot = plt.figure(figsize = (10,6))
        plt.hist(market_data[column], bins=50, color='green', edgecolor='black')
        st.pyplot(plot)


# Visulization on Campaings :
st.subheader("Visulization on each campaing result :")
cmp1 = market_data['AcceptedCmp1'].value_counts()
cmp2 = market_data['AcceptedCmp2'].value_counts()
cmp3 = market_data['AcceptedCmp3'].value_counts()
cmp4 = market_data['AcceptedCmp4'].value_counts()
cmp5 = market_data['AcceptedCmp5'].value_counts()

data = {'Not_Accept': [cmp1[0], cmp2[0], cmp3[0], cmp4[0], cmp5[0]],
         'Accept' : [cmp1[1], cmp2[1], cmp3[1], cmp4[1], cmp5[1]]}

campaign_data = pd.DataFrame(data, index = ['cmp1','cmp2','cmp3','cmp4','cmp5'])
st.dataframe(campaign_data)

st.bar_chart(campaign_data)


# Feature Engineering :
final_df = market_data.copy()

final_df['Education'] = final_df['Education'].map({'Basic' : 0, '2n Cycle' : 0 , 'Graduation':1, 'Master':2, 'PhD' : 3})

final_df['Marital_Status'] = final_df['Marital_Status'].map({'Married' : 2, 'Together': 2, 'Absurd' : 1, 'YOLO' : 1,
                                                             'Single':1, 'Divorced' : 1,'Widow' : 1, 'Alone' : 1})
final_df['Num_Kids'] = final_df.Kidhome.values + final_df.Teenhome.values

final_df['Fam_Size'] = final_df.Marital_Status.values + final_df.Num_Kids.values

final_df['Num_Accepted'] = final_df.AcceptedCmp1.values + final_df.AcceptedCmp2.values + \
                                final_df.AcceptedCmp3.values + final_df.AcceptedCmp4.values + \
                                final_df.AcceptedCmp5.values + final_df.Response.values

final_df['MntTotal'] = final_df['MntWines'].values + final_df['MntFruits'].values + \
                            final_df['MntMeatProducts'].values + final_df['MntFishProducts'].values + \
                            final_df['MntWines'].values + final_df['MntSweetProducts'].values + \
                            final_df['MntGoldProds'].values

final_df['Total_purchase'] = final_df['NumDealsPurchases'] + final_df['NumWebPurchases'] + \
                              final_df['NumCatalogPurchases'] + final_df['NumStorePurchases']

df = final_df.copy()

df.drop(['AcceptedCmp1', 'AcceptedCmp2','AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Kidhome','Total_purchase','Teenhome','Num_Kids',            'Marital_Status','Total_purchase'], axis=1, inplace=True)

final_df.drop(['AcceptedCmp1', 'AcceptedCmp2','AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Kidhome', 'Teenhome','NumDealsPurchases',
               'NumWebPurchases','NumCatalogPurchases','NumStorePurchases','Num_Kids', 'Marital_Status','MntTotal'], axis=1, inplace=True)



# FeatureScaling :
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(final_df), columns = final_df.columns)
st.subheader("Standardization of data : ")
st.dataframe(scaled_data)
scaled_datas = scaled_data.drop(['Age','Recency','Response','Complain','Education','Fam_Size','Num_Accepted'], axis = 1)


# Principal Componant Analysis
from sklearn.decomposition import PCA
pca = PCA()
pca.fit_transform(scaled_datas)
var = pca.explained_variance_ratio_
variance = np.cumsum(np.round(var, decimals = 4)*100)

plot = plt.figure(figsize = (10,6))
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
st.pyplot(plot)

pca = PCA(n_components = 4)
pca_data = pd.DataFrame(pca.fit_transform(scaled_datas))
st.subheader("PCA data :")
st.dataframe(pca_data)

from sklearn.cluster import KMeans
wcss = []
for i in range(1,11) :
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(pca_data)
    wcss.append(kmeans.inertia_)
    
plot = plt.figure(figsize = (10,6))
plt.plot(range(1,11) , wcss, color = 'black')
plt.scatter(range(1,11), wcss, color = 'red')
st.pyplot(plot)



# final model
kmeans = KMeans(n_clusters = 3)
kmeans.fit(pca_data)
labels = kmeans.labels_
df['Cluster_id'] = labels
print('Number of Customers in each Clusters : ')
print(df['Cluster_id'].value_counts())
print(" ")

st.dataframe(df.groupby('Cluster_id').mean())
final_data = df.copy()

#create new dataframes for each cluster
clus0 = final_data[final_data.Cluster_id == 0]
clus1 = final_data[final_data.Cluster_id == 1]
clus2 = final_data[final_data.Cluster_id == 2]

column_1 = ['Income','MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds','Recency',
            'Num_Accepted','NumWebPurchases','NumCatalogPurchases','Fam_Size','Age']
col = st.selectbox("select any options : ", options = column_1)


plot = plt.figure(figsize=(12,8))
sns.kdeplot(data=clus0, x=col, label='cluster 0')
sns.kdeplot(data=clus1, x=col, label ='Cluster 1')
sns.kdeplot(data=clus2, x=col, label ='Cluster 2')
plt.legend()
st.pyplot(plot)
    

    



import pickle

pickle_in = open("classifier.pkl", 'rb')
classifier = pickle.load(pickle_in)

# user Interface for Customer Behaviour Prediction

def prediction(Education, Marital_Status, kidhome, Teenhome, Income, Recency, MntWines, MntFruits, MntMeatProducts, 
               MntFishProducts,MntSweetProducts,MntGoldProducts,AcceptedCmp1,AcceptedCmp2, AcceptedCmp3,
               AcceptedCmp4, AcceptedCmp5, NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumStorePurchases, 
               NumWebVisitsMonth, Complain, Response, Age) :
    if (Education == 'Basic') | (Education == '2n Cycle') :
        Education = 0
    elif (Education == 'Graduation') :
        Education = 1
    elif (Education == 'Master') :
        Education = 2
    else : Education = 3
        
    if(Marital_Status == 'Married') | (Marital_Status == 'Together') :
        Marital_Status = 2
    else : Marital_Status = 1
        
    Fam_Size = Marital_Status + Kidhome + Teenhome
    Num_Accepted = AcceptedCmp1 + AcceptedCmp2 + AcceptedCmp3 + AcceptedCmp4 + AcceptedCmp5 + Response
    MntTotal = MntWines + MntFruits + MntMeatProducts + MntFishProducts + MntSweetProducts + MntGoldProducts
    
    prediction = classifier.predict([[Education, Income, Recency, MntWines, MntFruits, MntMeatProducts, MntFishProducts, 
                                            MntSweetProducts, MntGoldProducts, NumDealsPurchases, NumWebPurchases, NumCatalogPurchases,
                                            NumStorePurchases, NumWebVisitsMonth, Complain, Response, Age, Fam_Size, Num_Accepted, 
                                            MntTotal]])
    if prediction == 0 :
        return 'Economic Customer'
    elif prediction == 1 :
        return 'Normal Customer'
    else : return 'Premium Customer'



st.subheader("Customers Behaviour Prediction")
Education = st.selectbox("Education :", options = ['Basic','2n Cycle','Graduation','Master','PhD'])
Marital_Status = st.selectbox("Merital Status :", options = ['Married','Together','Absurd','YOLO','Single','Divorced','Widow','Alone'])
Kidhome = st.number_input("Number of Kid :", step = 1)
Teenhome = st.number_input("Number of Teen :", step = 1)
Income = st.number_input("Income :", step = 1)
Recency = st.number_input("Number of days last purchase :", step = 1)

MntWines = st.number_input("Amount spent on wine in last 2 years :", step = 1)
MntFruits = st.number_input("Amount spent on Fruits in last 2 years :", step = 1)
MntMeatProducts = st.number_input("Amount spent on Meat in last 2 years :", step = 1)
MntFishProducts = st.number_input("Amount spent on Fish in last 2 years :", step = 1)
MntSweetProducts = st.number_input("Amount spent on sweet in last 2 years :", step = 1)
MntGoldProducts = st.number_input("Amount spent on Gold in last 2 years :", step = 1)

AcceptedCmp1 = st.number_input("Response in 1st camp : 1 if yes & 0 if No ", step = 1)
AcceptedCmp2 = st.number_input("Response in 2nd camp : 1 if yes & 0 if No ", step = 1)
AcceptedCmp3 = st.number_input("Response in 3rd camp : 1 if yes & 0 if No ", step = 1)
AcceptedCmp4 = st.number_input("Response in 4th camp : 1 if yes & 0 if No ", step = 1)
AcceptedCmp5 = st.number_input("Response in 5th camp : 1 if yes & 0 if No ", step = 1)

NumDealsPurchases = st.number_input("Number of purchase with discount :", step = 1)
NumWebPurchases = st.number_input("Number of purchase through website :", step = 1)
NumCatalogPurchases = st.number_input("Number of purchase using Catalog :", step = 1)
NumStorePurchases = st.number_input("Number of purchase from store :", step = 1)

NumWebVisitsMonth = st.number_input("Number of visits to company’s website in the last month :", step = 1)
Complain = st.number_input("customer's complain : 1 if yes & 0 if No ", step = 1)
Response = st.number_input("Response in last camp : 1 if yes & 0 if No ", step = 1)
Age = st.number_input("Age : ", step = 1)
 
if st.button("Predict") :    
    result = prediction(Education,Marital_Status,Income,Kidhome,Teenhome, Recency, MntWines, MntFruits,MntMeatProducts, MntFishProducts, 
                        MntSweetProducts, MntGoldProducts,AcceptedCmp1,AcceptedCmp2, AcceptedCmp3, AcceptedCmp4, AcceptedCmp5,
                        NumDealsPurchases, NumWebPurchases, NumCatalogPurchases,NumStorePurchases, NumWebVisitsMonth,
                        Complain, Response, Age)
    st.write("The Customer belongs from the Cluster called : ", result)

    























