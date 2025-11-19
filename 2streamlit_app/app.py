import streamlit as st
import pandas as pd
import pickle as pk
from sklearn.preprocessing import PolynomialFeatures
import os

st.title("Expenses Prediction")

# 	age	sex	bmi	children	smoker	region
# input form
age = st.number_input("Enter age:", 16, 80)
sex = st.radio("Sex:", ["male", "female"])
bmi = st.number_input("Enter bmi:", 10, 60)
children = st.number_input("Enter no. of children:", 0, 10)
smoker = st.radio("Smoker:", ["yes", "no"])
region = st.selectbox("Region:", ["southwest", "southeast", "northwest", "northeast"])

if st.button("Submit"):
    if sex == "male":
        sex = True
    else:
        sex = False

    if smoker == "yes":
        smoker = True
    else:
        smoker = False

    if region == "southeast":
        se = 1
        sw = 0
        ne = 0
        nw = 0
    elif region == "southwest":
        se = 0
        sw = 1
        ne = 0
        nw = 0
    elif region == "northeast":
        se = 0
        sw = 0
        ne = 1
        nw = 0
    else:
        se = 0
        sw = 0
        ne = 0
        nw = 1
    # during training:"age","bmi","children","male","northeast","northwest","southeast","southwest","is_smoker"
    df = pd.DataFrame(
        {
            "age": [age],
            "bmi": [bmi],
            "children": [children],
            "male": [sex],
            "northeast": [ne],
            "northwest": [nw],
            "southeast": [se],
            "southwest": [sw],
            "is_smoker": [smoker],
        }
    )
    # st.dataframe(df)
    path = os.path.join(os.path.dirname(__file__), "bestmodel.pickle")
    dbfile = open("bestmodel.pickle", "rb")
    model = pk.load(dbfile)
    p = PolynomialFeatures(degree=2)
    result = model.predict(p.fit_transform(df))[0][0]
    # print(result)
    st.write("The predicted expenses = ", round(result, 2))
