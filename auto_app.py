import streamlit as st
import pickle
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder

st.sidebar.title('Car Price Prediction')
html_temp = """
<div style="background-color:blue;padding:10px">
<h2 style="color:white;text-align:center;">Streamlit ML Cloud App </h2>
</div>"""
st.markdown(html_temp, unsafe_allow_html=True)

st.write("Hello World! , This is some Data:")

df=pd.read_csv("final_scout_not_dummy.csv")

st.table(df.head(3))

