from fastapi import FastAPI
import pandas as pd
import pickle as pk
from sklearn.preprocessing import PolynomialFeatures

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


# age	sex	bmi	children	smoker	region
@app.get("/expenses/")
def read_item(age: int, sex: str, bmi: float, children: int, smoker: str, region: str):
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
    dbfile = open("bestmodel.pickle", "rb")
    model = pk.load(dbfile)
    p = PolynomialFeatures(degree=2)
    result = model.predict(p.fit_transform(df))[0][0]
    return {"predicted_expenses": round(result, 2)}
