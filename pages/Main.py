import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import re
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.utils import compute_sample_weight


st.title('Predictive Maintenance with Engine Sensor Data')
st.header('')
st.header('')
st.write('Predictive maintenance, also referred to as condition-based maintenance, involves performance '
         'monitoring and equipment condition monitoring during regular operations to reduce the chances'
         ' of a breakdown. Manufacturers began using predictive maintenance in the nineties.')

st.write('Predictive maintenance’s main goal is to predict equipment failures based on certain parameters '
         'and factors. Once predicted, manufacturers take needed steps to prevent this failure '
         'with corrective or scheduled maintenance.')

st.write('This project in particular will use 10 000 rows of data from boat engine sensors to train '
         'a model that will help predict failures. ')
st.header('')

columns = ["UDI", "Product ID", "Type", "Air Temperature [K]", "Process Temperature [K]", "Rotational Speed [rpm]",
           "Torque [Nm]", "Tool Wear [min]", "Target", "Failure Type"]

cols = ["Type", "Air Temperature [°C]", "Process Temperature [°C]", "Rotational Speed [rpm]", "Torque [Nm]",
        "Tool Wear [min]", "Target", "Failure Type"]


data = pd.read_csv("predictive_maintenance.csv", sep=",", names=columns)


st.header('')
st.header('Exploratory Data Analysis')
st.header('')
st.subheader('Original Data')
st.header('')
st.write(data)
st.header('')

data.drop(index=data.index[0], axis=0, inplace=True)
data.drop(columns=['Product ID', 'UDI'], inplace=True)
data['Air Temperature [K]'] = data['Air Temperature [K]'].astype('float')
data['Process Temperature [K]'] = data['Process Temperature [K]'].astype('float')
data['Rotational Speed [rpm]'] = data['Rotational Speed [rpm]'].astype('float')
data['Torque [Nm]'] = data['Torque [Nm]'].astype('float')
data['Tool Wear [min]'] = data['Tool Wear [min]'].astype('float')


start_date = pd.to_datetime('2022-01-01 00:00:00')
time_delta = pd.to_timedelta(2, unit='m')
data['DateTime'] = start_date + (np.arange(len(data)) * time_delta)

data["Air Temperature [K]"] = data["Air Temperature [K]"] - 272.15
data["Process Temperature [K]"] = data["Process Temperature [K]"] - 272.15
data.rename(columns={"Air Temperature [K]": "Air Temperature [°C]",
                     "Process Temperature [K]": "Process Temperature [°C]"}, inplace=True)


st.write('Some adjustments needed to be made to the data including removing the columns containing the UDI and '
         'Product ID, converting air temperatures and process temperatures from Kelvin to Celsius, finding their '
         'difference as well as creating a DateTime column to convert the data into hourly measurements. '
         'Based on the DateTime column another column has been added containing the time elapsed since last '
         'failure at the time of the measurement.')

st.header('')
st.header('')
st.subheader('Dataframe preview')
st.header('')
st.dataframe(data)
st.header('')
st.header('')


with st.expander('Unique values per column and their counts'):

    st.header('')

    for col in cols:
        st.subheader(col)
        st.write(data[col].value_counts())

st.header('')
st.header('')
with st.expander('Percentage of zero values in columns'):

    st.header('')

    st.write(data[data == '0'].count(axis=0)/len(data.index)*100)
    st.caption('')

st.header('')
st.header('')
st.write('Checking for rows that have Target = 1 but Failure Type = No Failure')
df_failure = data[data['Target'] == '1'].copy()
st.write(df_failure['Failure Type'].value_counts())
st.header('')
st.write('It appears that 9 rows are mislabeled, they will be removed')
st.write(df_failure[df_failure['Failure Type'] == 'No Failure'])

index_possible_failure = df_failure[df_failure['Failure Type'] == 'No Failure'].index
data.drop(index_possible_failure, axis=0, inplace=True)

st.header('')
st.header('')
st.write('Checking for rows that have Target = 0 but Failure Type != No Failure')
df_failure = data[data['Target'] == '0'].copy()
st.write(df_failure['Failure Type'].value_counts())
st.header('')

st.write('It appears that 18 rows are mislabeled, they will be removed')
st.write(df_failure[df_failure['Failure Type'] == 'Random Failures'])


index_possible_failure = df_failure[df_failure['Failure Type'] == 'Random Failures'].index
data.drop(index_possible_failure, axis=0, inplace=True)

data.reset_index(inplace=True, drop=True)

st.header('')
st.header('')


data_numeric = data[cols].copy()
st.subheader('Additional information about the data in each column')
st.header('')
st.dataframe(data_numeric.describe().style.background_gradient(cmap='bone'))

st.header('')
st.header('')
st.write("--------------------------------------------------------")
st.header('')


st.header('Visualisations')
st.header('')

with st.expander('Bar charts of values for all features'):

    px.histogram(data, y="Failure Type", color="Failure Type")

    plot = sns.displot(data=data, x="Air Temperature [°C]", kde=True, bins=100, height=5,
                       aspect=3.5)
    st.pyplot(plot)
    plot = sns.displot(data=data, x="Process Temperature [°C]", kde=True, bins=100, height=5,
                       aspect=3.5)
    st.pyplot(plot)

    plot = sns.displot(data=data, x="Rotational Speed [rpm]", kde=True, bins=100,
                       height=5, aspect=3.5)
    st.pyplot(plot)
    plot = sns.displot(data=data, x="Torque [Nm]", kde=True, bins=100,
                       height=5, aspect=3.5)
    st.pyplot(plot)
    plot = sns.displot(data=data, x="Tool Wear [min]", kde=True, bins=100,
                       height=5, aspect=3.5)
    st.pyplot(plot)


st.header('')
st.header('')

with st.expander('Boxplots for all features against Failure Types'):
    def create_histogram(column_name):
        plt.figure(figsize=(16, 6))
        return px.box(data_frame=data, y=column_name, color='Failure Type', points="all")


    st.write(create_histogram('Air Temperature [°C]'))
    st.write(create_histogram('Process Temperature [°C]'))
    st.write(create_histogram('Rotational Speed [rpm]'))
    st.write(create_histogram('Torque [Nm]'))
    st.write(create_histogram('Tool Wear [min]'))

st.header('')
st.header('')

plot = sns.pairplot(data, height=2.5, hue='Failure Type')
st.pyplot(plot)

st.header('')
st.header('')
st.write("--------------------------------------------------------")
st.header('')

st.write('Failure percentage in data')
st.write(data['Target'].value_counts(normalize=1) * 100)
st.header('')
st.write('The data appears to be extremely unbalanced in regards to the amount of recorded failures so it will need '
         'to be oversampled')
st.write('In order to avoid a data leak we will have to remove either Target or Failure Types. I have chosen to keep'
         ' Failure types as it provides more detailed information on the nature of the failures.')

data_numeric.drop(columns=['Target'], inplace=True)
regex = re.compile(r"\[|\]|<", re.IGNORECASE)

data_numeric.columns = [regex.sub("_", col) if any(x in str(col) for x in {'[', ']', '<'})
                  else col for col in data_numeric.columns.values]

dict1 = {'L': 0, 'M': 1, 'H': 2}
data_numeric['Type'] = [dict1.get(x, -1) for x in data_numeric['Type']]

dict = {'No Failure': 0, 'Heat Dissipation Failure': 1, 'Power Failure': 2, 'Overstrain Failure': 3, 'Tool Wear Failure': 4, 'Random Failures' : 5}
data_numeric['Failure Type'] = [dict.get(x, -1) for x in data_numeric['Failure Type']]

st.header('')
st.write(data_numeric)

st.header('')

scaler = LabelEncoder()
data_numeric['Failure Type'] = scaler.fit_transform(data_numeric['Failure Type'])
X = data_numeric.drop(columns='Failure Type', axis=1)
y = data_numeric['Failure Type']
X_data, X_test, y_data, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

weight_train = compute_sample_weight('balanced', y_data)
weight_test = compute_sample_weight('balanced', y_test)


st.write("--------------------------------------------------------")

st.header('Training models and comparing results')
st.header('')

# XGBoost

xgb = XGBClassifier(booster='gbtree',
                    sampling_method='gradient_based',
                    eval_metric='aucpr',
                    objective='multi:softmax',
                    num_class=6
                    )

xgb.fit(X_data, y_data, sample_weight=weight_train)
y_pred_xgb = xgb.predict(X_test)

xgb_data = round(xgb.score(X_data, y_data) * 100, 2)
xgb_accuracy = round(accuracy_score(y_pred_xgb, y_test) * 100, 2)

st.subheader('XGBoost')
st.write('Training Accuracy    :', xgb_data, '%')
st.write('Model Accuracy Score :', xgb_accuracy, '%')
s = 'Classification_Report: \n' + classification_report(y_test, y_pred_xgb)
st.text(s)
st.header('')

cm = confusion_matrix(y_test, y_pred_xgb, labels=xgb.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=xgb.classes_)
disp.plot(cmap='magma')
st.pyplot(plt)
st.write('--------------------------------------------------------')


# Logistic Regression
logreg = LogisticRegression(random_state=8)
logreg.fit(X_data, y_data, sample_weight=weight_train)
y_pred_lr = logreg.predict(X_test)

log_data = round(logreg.score(X_data, y_data) * 100, 2)
log_accuracy = round(accuracy_score(y_pred_lr, y_test) * 100, 2)

st.subheader('Logistic Regression')
st.write('Training Accuracy    :', log_data, '%')
st.write('Model Accuracy Score :', log_accuracy, '%')
s = 'Classification_Report: \n' + classification_report(y_test, y_pred_lr)
st.text(s)
st.header('')

cm = confusion_matrix(y_test, y_pred_lr, labels=logreg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=logreg.classes_)
disp.plot(cmap='magma')
st.pyplot(plt)
st.write('--------------------------------------------------------')


# Decision Tree
decision = DecisionTreeClassifier(random_state=8)
decision.fit(X_data, y_data, sample_weight=weight_train)
y_pred_dec = decision.predict(X_test)

decision_data = round(decision.score(X_data, y_data) * 100, 2)
decision_accuracy = round(accuracy_score(y_pred_dec, y_test) * 100, 2)

st.subheader('Decision Tree Classifier')
st.write('Training Accuracy    :', decision_data, '%')
st.write('Model Accuracy Score :', decision_accuracy, '%')
s = 'Classification_Report: \n' + classification_report(y_test, y_pred_dec)
st.text(s)
st.header('')

cm = confusion_matrix(y_test, y_pred_dec, labels=decision.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=decision.classes_)
disp.plot(cmap='magma')
st.pyplot(plt)
st.write('--------------------------------------------------------')


# Random Forest
random_forest = RandomForestClassifier(n_estimators=100, random_state=8)
random_forest.fit(X_data, y_data, sample_weight=weight_train)
y_pred_rf = random_forest.predict(X_test)
random_forest.score(X_data, y_data)

random_forest_data = round(random_forest.score(X_data, y_data) * 100, 2)
random_forest_accuracy = round(accuracy_score(y_pred_rf, y_test) * 100, 2)

st.subheader('Random Forest Classifier')
st.write('Training Accuracy    :', random_forest_data, '%')
st.write('Model Accuracy Score :', random_forest_accuracy, '%')
s = 'Classification_Report: \n' + classification_report(y_test, y_pred_rf)
st.text(s)
st.header('')

cm = confusion_matrix(y_test, y_pred_rf, labels=random_forest.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=random_forest.classes_)
disp.plot(cmap='magma')
st.pyplot(plt)
st.write('--------------------------------------------------------')

# Support Vector Machines
svc = SVC(random_state=8)
svc.fit(X_data, y_data, sample_weight=weight_train)
y_pred_svc = svc.predict(X_test)

svc_data = round(svc.score(X_data, y_data) * 100, 2)
svc_accuracy = round(accuracy_score(y_pred_svc, y_test) * 100, 2)

st.subheader('Support Vector Machines')
st.write('Training Accuracy    :', svc_data, '%')
st.write('Model Accuracy Score :', svc_accuracy, '%')
s = 'Classification_Report: \n' + classification_report(y_test, y_pred_svc)
st.text(s)
st.header('')

cm = confusion_matrix(y_test, y_pred_svc, labels=svc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=svc.classes_)
disp.plot(cmap='magma')
st.pyplot(plt)
st.write('--------------------------------------------------------')


models = pd.DataFrame({
    'Model': [
        'XGBoost', 'Logistic Regression', 'Decision Tree',  'Random Forest', 'Support Vector Machines',
    ],

    'Training Accuracy':
        [xgb_data, log_data, decision_data, random_forest_data, svc_data],

    'Model Accuracy Score': [
        xgb_accuracy, log_accuracy, decision_accuracy, random_forest_accuracy,  svc_accuracy ]
})

st.header('')
st.subheader('Comparing the models')
st.header('')
pd.set_option('display.precision', 2)
models.sort_values(by='Model Accuracy Score', ascending=False).style.background_gradient(
       cmap='coolwarm').hide_index().set_properties(**{
            'font-family': 'Lucida Calligraphy',
            'color': 'LightGreen',
            'font-size': '15px'
        })

st.dataframe(models)

st.header('')
st.header('')
st.write('Choosing Random Forest because it has the highest model accuracy score of 98.2%')
st.header('')
st.header('')
st.subheader('Most important features for Random Forest Model:')

importance = random_forest.feature_importances_
feature_ranking = sorted(zip(importance, X_data.columns), reverse=True)
for i, (score, name) in enumerate(feature_ranking):
    st.write("%d. Feature '%s' (%.2f%%)" % (i + 1, name, score*100))

prediction1 = random_forest.predict(X_test)

cross_checking = pd.DataFrame({'Actual': y_test, 'Predicted': prediction1})
cross_checking.sample(5).style.background_gradient(
        cmap='coolwarm').set_properties(**{
            'font-family': 'Lucida Calligraphy',
            'color': 'LigntGreen',
            'font-size': '15px'
        })

st.header('')
st.header('')
st.subheader('Actual vs Predicted')
st.header('')
st.write(cross_checking.style.background_gradient(cmap='Purples'))

unique_failure_types = data['Failure Type'].unique().tolist()
unique_failure_types.remove('No Failure')
for failure_type in unique_failure_types:
    data[f'RUL_{failure_type}'] = -1

data_numeric2 = data_numeric.copy()
data_numeric2.drop(columns=['Failure Type'], inplace=True)
pred = random_forest.predict(data_numeric2.values)

reverse_dict = {v: k for k, v in dict.items()}
pred = [reverse_dict.get(value, -1) for value in pred]


rul_data = data.copy()

rul_data['Failure Type Prediction'] = pred


# Initialize the RUL columns to -1
for failure_type in unique_failure_types:
    rul_data[f'RUL_{failure_type}'] = -1

for i in range(len(rul_data)):
    row = rul_data.iloc[i]
    for failure_type in unique_failure_types:
        next_failure_indices = rul_data[
            (rul_data['Failure Type Prediction'] == failure_type) & (rul_data.index > i)].index.tolist()

        if len(next_failure_indices) == 0:
            next_failure_indices = rul_data[
                (rul_data['Failure Type Prediction'] == failure_type) & (rul_data.index <= i)].index.tolist()

        if len(next_failure_indices) == 0:
            if rul_data.loc[i, f'RUL_{failure_type}'] == -1:
                rul_data.loc[i, f'RUL_{failure_type}'] = -1
            else:
                rul_data.loc[i, f'RUL_{failure_type}'] = (rul_data.index[-1] - i) % len(rul_data)
        else:
            next_failure_index = next_failure_indices[0]
            time_until_failure = (next_failure_index - i) % len(rul_data)
            rul_data.loc[i, f'RUL_{failure_type}'] = time_until_failure

st.header('Creating RUL predictions for all failure types using the trained model:')
st.header('')
st.write(rul_data)

# rul_data.to_csv('rul_data.csv', index=False)
