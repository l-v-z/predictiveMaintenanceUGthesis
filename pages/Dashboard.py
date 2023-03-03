import streamlit as st
import time
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Real-Time Data Dashboard", page_icon="Active", layout="wide")

columns = ["UDI", "Product ID", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
           "Target", "Failure Type"]
df = pd.read_csv("predictive_maintenance.csv", sep=",", names=columns)

job_filter = st.selectbox("Select Failure Type", pd.unique(df["Failure Type"]))

kpi1, kpi2 = st.columns(2)

# fill the column with respect to the KPIs
kpi1.metric(label="Failure probability", value=round(20.66), delta=round(20.66) - 10)


kpi2.metric(label="Health", value=int(20.66), delta=10 + 20.66)


# create columns for the chars
fig_col1, fig_col2 = st.columns(2)

with fig_col1:
    st.markdown("Chart 1")
    fig1 = px.density_heatmap(data_frame=df, y="Failure Type", x="Torque [Nm]")
    st.write(fig1)

with fig_col2:
    st.markdown("Chart 2")
    fig2 = px.histogram(data_frame=df, x="Process temperature [K]")
    st.write(fig2)

