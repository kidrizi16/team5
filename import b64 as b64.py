import b64 as b64
import conversions as conversions
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import seaborn as sns
import matplotlib.pyplot as plt

df_selected = pd.read_csv('C:/Users/tocil/Downloads/campaignanalytics2.csv')
df_selected_all = df_selected[['Region'],['Country'],['Product_Category'],['Campaign_Name'],['Revenue'],['Revenue_Target'],['City'],['State']].copy()

#Define a function that allows us to download the read-in data
def filedownload(df):
    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes

    conversions
    href = f'<a href="data:file/csv;base64,{b64}" download = "campaign_data.csv">Download CSV File</a>'
    return href

st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown(filedownload(df_selected_all), unsafe_allow_html=True)
"""
header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
model_training = st.beta_container()

with header:
    st.title("Welcome to project no 3 of team no 5")
    st.text("In this project we are going to analyze campaigns")

with dataset:
    st.header("Campaign analytics dataset")
    st.text("This dataset was given to us by our instructor")

    campaign_data = pd.read_csv('C:/Users/tocil/Downloads/campaignanalytics2.csv')
    st.write(campaign_data.head())

    st.subheader('Pick up revenue distribution')
    our_data = pd.DataFrame(campaign_data['Revenue'].value_counts()).head(50)
    st.bar_chart(our_data)

with features:
    st.header("The features we created")

with model_training:
    st.header("Time to train the model!")
    st.text("Here you get to choose the hyperparameters of the model and see how the performance changes.")
"""