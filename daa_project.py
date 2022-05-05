import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import plotly.express as px

class A():
    def teamA(self):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        df=pd.read_csv("data//DAA_DATA.csv")
        st.title("Placement Score Predictor")
        st.image("data//pic.png")

        st.header("Choose a graph to visualize the data")

        graph = st.selectbox(" ",["Interactive","Non-Interactive"])

        val = st.slider("Filter data using years",0.00,10.00)
        df = df.loc[df["cgpa"]>= val]
        if graph == "Non-Interactive":
            plt.figure(figsize = (10,5))
            plt.scatter(df["cgpa"],df["placement_score"])
            plt.ylim(0)
            plt.xlabel("CGPA")
            plt.ylabel("Placement Score")
            plt.tight_layout()
            st.pyplot()
        if graph == "Interactive":

            fig=px.scatter(df, x="cgpa",y="placement_score",size_max=200,range_x=[0,10],range_y=[0,100])
            fig.update_layout(width=700)
            st.write(fig)



        st.header("Predict Your Placement Score")
        INPUT=st.number_input("Enter your CGPA",0.00,10.00)

        
        x=df.drop('placement_score',axis='columns')
        y=df.drop('cgpa',axis='columns')
        reg=LinearRegression()
        reg.fit(df[['cgpa']],df.placement_score)
        predi=reg.predict([[INPUT]])
        if st.button("Predict"):
            st.success(f"Your predicted Placement Score is {predi}")


    def main(self):
        self.teamA()

obj=A()
obj.teamA()
