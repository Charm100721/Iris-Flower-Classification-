#CSS 
css_content = """
<style>
[data-testid="stHeader"]{
    background: rgb(31,31,31);
}

[data-testid="stSidebar"]{
    background: rgb(75,74,83);
}

[data-testid="stAppViewContainer"]{
    background: rgb(0,0,0);
}


</style>

"""


#IMPORTING LIBRARIES
from turtle import onclick
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#PRINTING DIFFERENT DATASETS IN SCIKITLEARN
print(dir(datasets))

#EXPLORING THE DATA OF LOAD_IRIS
from sklearn.datasets import load_iris
data=load_iris()
print(data["feature_names"])
print(data["data"])
print(data["target_names"])
print(data["target"])

#CONVERTING THE ARRAYS TO DATAFRAME
iris_data = pd.DataFrame(data["data"], columns=data["feature_names"])
iris_data["Result"] = data["target"]
print(iris_data)
print(iris_data.shape)
print(iris_data.isnull().sum())


#TRAINING THE MODEL
    # X = iris_data.iloc[:, 0:4]
    # Y = iris_data["Result"]

    # X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size= 0.20, random_state=1)
    # print(X.shape, X_train.shape, X_test.shape)
    # print(Y.shape, Y_train.shape, Y_test.shape)

    # model = LogisticRegression(random_state=1)
    # model.fit(X_train, Y_train)

#MAKE A PREDICTION MODEL
    # y_pred = model.predict(X_test)

#CHECK FOR THE ACCURACY & CONFUSION MATRIX OF THE MODEL
    # plt.figure(figsize=(3,3))
    # sns.heatmap(confusion_matrix(Y_test, y_pred), annot=True)
    # plt.xlabel("Predicted Values")
    # plt.ylabel("Actual Values")
    # plt.show()

    # print(accuracy_score(Y_test, y_pred))

#PRINTING THE MINIMUM AND MAXIMUM VALUE OF EACH FEATURES FOR SLIDER
print(iris_data["sepal length (cm)"].max(), iris_data["sepal length (cm)"].min())
print(iris_data["sepal width (cm)"].max(), iris_data["sepal width (cm)"].min())
print(iris_data["petal length (cm)"].max(), iris_data["petal length (cm)"].min())
print(iris_data["petal width (cm)"].max(), iris_data["petal width (cm)"].min())



#CREATING A TITLE
st.markdown(css_content, unsafe_allow_html=True)
st.title("""
Iris Flower Prediction App
This app predicts **iris flower** type!
""")

display_df = st.checkbox("Display Dataframe")

if display_df == True:
    st.dataframe(iris_data)

#CREATING A TITLE FOR THE SIDEBAR
st.sidebar.header("**Input Features**")


#SETTING THE SIDEBARS IN EACH FEATURES AND STORING IT IN A DATAFRAME
def input_user_features():
    sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.3, 7.9, 6.7)
    sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.4, 3.0)
    petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 6.9, 5.2)
    petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 2.3)
    New_dataset = {
                    "Sepal Length" : sepal_length,
                    "Sepal Width" : sepal_width,
                    "Petal Length" : petal_length,
                    "Petal Width" : petal_width
    }
    features = pd.DataFrame(New_dataset, index=[0])
    return features

df = input_user_features()

st.subheader("Summary of Input Features")
st.write(df)



#TRAINING THE MODEL AND MAKE A PREDICTION
X = iris_data.iloc[:, 0:4]
Y = iris_data["Result"]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size= 0.20)
print(X.shape, X_train.shape, X_test.shape)
print(Y.shape, Y_train.shape, Y_test.shape)

model = RandomForestClassifier()
model.fit(X_train, Y_train)
y_pred = model.predict(df)

def predict():
    st.subheader("Prediction")
    result = data.target_names[y_pred]
    st.success(f"The Iris Flower Type is **{result[0]}**.")

    if result[0] == "setosa":
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/Iris_setosa_2.jpg/1130px-Iris_setosa_2.jpg")
    elif result[0] == "versicolor":
        st.image("https://kiefernursery.com/wp-content/uploads/2020/08/Iris-Purple-Flame-e1597336517307.jpg")
    else:
        st.image("https://www.fs.usda.gov/wildflowers/beauty/iris/Blue_Flag/images/iris_virginica/iris_virginica_virginica_lg.jpg")

    print(data["target_names"])

    #PRINT THE PREDICTION PROBABILITY
    P_proba = model.predict_proba(df)
    print(P_proba)
    st.subheader("Percent Accuracy")
    st.write(f"For setosa:  {(P_proba[0][0])*100}%")
    st.write(f"For versicolor:  {(P_proba[0][1])*100}% ")
    st.write(f"For virginica:  {(P_proba[0][2])*100}% ")

p_button = st.button("Predict!", help="Click to predict!")

if p_button == True:
    predict()

