# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 07:36:13 2023

@author: tejas
"""

import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import pickle

st.title ('Model Deployment: KNN')

st.sidebar.header('User Input Parameters')



def user_input_parameters():
    
    cus_education = st.sidebar.selectbox('Education of the Cusotmer', ('0','1','2','3','4'))
    cus_income = st.sidebar.number_input('Income of Customer', step = 1)
    cus_recency = st.sidebar.number_input('Days after recent perchase', step = 1)
    cus_wines = st.sidebar.number_input('Money spent on Wines', step = 1)
    cus_fruits = st.sidebar.number_input('Money spent on Fruits', step = 1)
    cus_meat = st.sidebar.number_input('Money spent on Meat', step = 1)
    cus_fish = st.sidebar.number_input('Money spent on Fish', step = 1)
    cus_sweet = st.sidebar.number_input('Money spent on sweet', step = 1)
    cus_gold = st.sidebar.number_input ('Money spent on Gold products', step = 1)
    cus_ol_purchase = st.sidebar.number_input('No. of online purchases', step = 1)
    cus_visits = st.sidebar.number_input('No. of times customer visits website', step = 1)
    cus_month = st.sidebar.number_input ('Customer month from registering', step = 1)
    cus_age = st.sidebar.number_input('Age of Customer', step = 1)
    cus_child = st.sidebar.selectbox('No of children',('1', '2', '3', '4', '5') )
    cus_isparent = st.sidebar.selectbox('Is customer a Parrent?', ('1','0'))
    cus_family_size = st.sidebar.selectbox ('No. of family members', ('1','2','3','4', '5'))
    cus_accep = st.sidebar.selectbox('No of campaigns accepted',('0','1', '2', '3', '4', '5'))
    cus_off_purchase = st.sidebar.number_input('No. of offline purchases', step = 1)
    
    data = {'Education':cus_education,
            'Income':cus_income,
            'Recency':cus_recency,
            'Wines':cus_wines,
            'Fruits':cus_fruits,
            'Meat':cus_meat,
            'Fish':cus_fish,
            'Sweet':cus_sweet,
            'Gold':cus_gold,
            'Online purchase':cus_ol_purchase,
            'Customer_web_visits':cus_visits,
            'Month Customer':cus_month,
            'Age':cus_age,
            'Children':cus_child,
            'Is Parent':cus_isparent,
            'Family size':cus_family_size,
            'Campaigns Accepted':cus_accep,
            'Offline purchase':cus_off_purchase,}
    feature = pd.DataFrame(data, index =[0])
    return feature
df = user_input_parameters()
st.write(df)
loaded_model = pickle.load(open('C:\\Users\\tejas\\DS_Project_CPA\\Customer_clusters_model.sav', 'rb'))

if st.button("Predict the Customer type"):
     predictions = loaded_model.predict(df)
     if predictions == 0:
         st.write ('''The customer belongs to Cluster 1 ''')
         st.write ( '''Characteristics :

                   - Belongs to Low Income Class
                   - Childrens between the range (0,3)
                   - Majority of them are parents
                   - At max there are 5 members in the family and at least 1''')
     elif predictions == 1:
         st.write ('''The customer belongs to Cluster 2''')
         st.write ('''Characteristics :
                   
                   - Belongs to High Income Class
                   - Childrens between the range (0,1)
                   - Majority of them are not parents
                   - At max there are 3 members in the family and at least 1''')
     elif predictions == 2:
         st.write ('''The customer belongs to Cluster 3''')
         st.write ('''Characteristics :
                   
                   - Belongs to High Middle Class
                   - Childrens between the range (0,1)
                   - Majority of them are not parents
                   - At max there are 3 members in the family and at least 1''')
                   
     elif predictions == 3:
         st.write ('''The customer belongs to Cluster 4''')
         st.write ('''Characteristics :
                   - Belongs to Low Middle Class
                   - Childrens between the range (1,3)
                   - Are definitely a parents
                   - At max there are 5 members in the family and at least 2''')

    
    