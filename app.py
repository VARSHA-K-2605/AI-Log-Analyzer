import streamlit as st
import pandas as pd

st.title("AI-Based Log Analyzer Dashboard")

df = pd.read_csv("logs.csv")

st.subheader("Log Data")
st.dataframe(df)

st.subheader("Severity Counts")
st.bar_chart(df['severity'].value_counts())