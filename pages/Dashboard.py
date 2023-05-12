import time
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Real-Time Data Dashboard", page_icon="Active", layout="wide")

columns = ["Type", "Air Temperature [°C]", "Process Temperature [°C]", "Rotational Speed [rpm]", "Torque [Nm]",
           "Tool Wear [min]", "Target", "Failure Type", "DateTime", "Failure Type Prediction", "RUL_Power Failure",
           "RUL_Tool Wear Failure", "RUL_Overstrain Failure", "RUL_Heat Dissipation Failure"]

cols_num = ["Air Temperature [°C]", "Rotational Speed [rpm]", "Torque [Nm]", "Tool Wear [min]", "RUL_Power Failure",
            "RUL_Tool Wear Failure", "RUL_Overstrain Failure", "RUL_Heat Dissipation Failure"]


df = pd.read_csv("rul_data.csv", sep=",", names=columns)
df.drop(index=df.index[0], axis=0, inplace=True)

total_rows = len(df)
matching_rows = len(df[df['Failure Type'] == df['Failure Type Prediction']])
percentage = (matching_rows / total_rows) * 100

# st.write(f"Percentage of rows where Failure Type and Failure Type Prediction have the same value: {percentage}%")


current_torque = 0
current_tool_wear = 0
current_rot_speed = 0
current_air_temp = 0
next_pow_fail = 0
next_tool_wear_fail = 0
next_overstr_fail = 0
next_heat_diss_fail = 0

df[cols_num] = df[cols_num].apply(pd.to_numeric)


placeholder = st.empty()

for i, row in df.iterrows():

    current_torque = df.loc[i, 'Torque [Nm]']
    current_tool_wear = df.loc[i, 'Tool Wear [min]']
    current_rot_speed = df.loc[i, 'Rotational Speed [rpm]']
    current_air_temp = round(df.loc[i, 'Air Temperature [°C]'], 2)
    next_pow_fail = df.loc[i, 'RUL_Power Failure']
    next_tool_wear_fail = df.loc[i, 'RUL_Tool Wear Failure']
    next_overstr_fail = df.loc[i, 'RUL_Overstrain Failure']
    next_heat_diss_fail = df.loc[i, 'RUL_Heat Dissipation Failure']

    with placeholder.container():

        n1, n2, n3, n4 = st.columns(4)

        n1.metric(
        label="Current Torque [Nm]",
        value=current_torque,
        delta=round(current_torque - df['Torque [Nm]'].mean()),
        )
        n2.metric(
        label="Current Tool Wear [min]",
        value=current_tool_wear,
        delta=round(current_tool_wear - df['Tool Wear [min]'].mean()),
        )

        n3.metric(
        label="Current Rotational Speed [rpm] ",
        value=current_rot_speed,
        delta=round(current_rot_speed - df['Rotational Speed [rpm]'].mean()),
        )

        n4.metric(
        label="Current Air Temperature [°C] ",
        value=current_air_temp,
        delta=round(current_air_temp - df['Air Temperature [°C]'].mean()),
        )

        st.header('')
        st.divider()
        st.header('')

        m1, m2, m3, m4 = st.columns(4)

        m2.metric(
        label="Next Power Failure (min)",
        value=next_pow_fail
        )

        m4.metric(
        label="Next Tool Wear Failure (min)",
        value=next_tool_wear_fail
        )

        m3.metric(
        label="Next Overstrain Failure (min)",
        value=next_overstr_fail
        )

        m1.metric(
        label="Next Heat Dissipation Failure (min) ",
        value=next_heat_diss_fail
        )

        time.sleep(2)

        st.header('')
        st.divider()
        st.header('')

        fig = px.line(df['Torque [Nm]'].iloc[:i + 1], title='Torque over Time')
        fig.update_traces(line_color='blue')
        st.plotly_chart(fig, use_container_width=True)

        st.header('')
        st.divider()
        st.header('')

        # fig = px.line(df['Tool Wear [min]'].iloc[:i + 1], title='Tool Wear over Time')
        # fig.update_traces(line_color='purple')
        # st.plotly_chart(fig, use_container_width=True)
        #
        # st.header('')
        # st.divider()
        # st.header('')

        fig = px.line(df['Rotational Speed [rpm]'].iloc[:i + 1], title='Rotational Speed over Time')
        fig.update_traces(line_color='yellow')
        st.plotly_chart(fig, use_container_width=True)

        st.header('')
        st.divider()
        st.header('')

        fig = px.line(df['Air Temperature [°C]'].iloc[:i + 1], title='Air Temperature °C over Time')
        fig.update_traces(line_color='red')
        st.plotly_chart(fig, use_container_width=True)

    placeholder.empty()




