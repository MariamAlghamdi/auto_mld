import streamlit as st
import pickle
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn import *

st.sidebar.title('Car Price Prediction')
html_temp = """
<div style="background-color:blue;padding:10px">
<h2 style="color:white;text-align:center;">Streamlit ML Cloud App </h2>
</div>"""
st.markdown(html_temp, unsafe_allow_html=True)

car_model=st.sidebar.selectbox("Select model of your car", ('Audi A1', 'Audi A3', 'Opel Astra', 'Opel Corsa', 'Opel Insignia', 'Renault Clio', 'Renault Duster', 'Renault Espace'))
vat=st.sidebar.radio('Select vat type',('VAT deductible','Price negotiable'))
km=st.sidebar.slider("What is the km of your car", 0,317000, step=1000)
Type=st.sidebar.radio('Select Used type',('Used',"Employee's car",'New', 'Demonstration', 'Pre-registered'))
Fuel=st.sidebar.radio('Select Fuel type',('Diesel','Benzine','LPG/CNG', 'Electric'))
age=st.sidebar.selectbox("What is the age of your car:",(0,1,2,3))
Upholstery_type=st.sidebar.radio('Select Upholstery_type type',('Cloth','Part/Full Leather'))
gearing_type=st.sidebar.radio('Select gear type',('Automatic','Manual','Semi-automatic'))

ds13_model=pickle.load(open("final_model_new","rb"))
ds13_transformer = pickle.load(open('transformer', 'rb'))



my_dict = {
    "make_model": car_model,
    "vat": vat,
    "km": km,
    "Type": Type,
    "Fuel": Fuel,
    "age": age,
    "Upholstery_type": Upholstery_type,
    "Gearing_Type": gearing_type
}


df = pd.DataFrame.from_dict([my_dict])


st.header("The configuration of your car is below")
st.table(df)

df2 = ds13_transformer.transform(df)

st.subheader("Press predict if configuration is okay")

if st.button("Predict"):
    prediction = ds13_model.predict(df2)
    st.success("The estimated price of your car is €{}. ".format(int(prediction[0])))
