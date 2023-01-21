import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, plot_confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


st.title('Predictive Maintenance with Engine Sensor Data')
st.caption('Predictive maintenance, also referred to as condition-based maintenance, involves performance '
           'monitoring and equipment condition monitoring during regular operations to reduce the chances'
           ' of a breakdown. Manufacturers began using predictive maintenance in the nineties.')

st.caption('Predictive maintenance’s main goal is to predict equipment failures based on certain parameters '
           'and factors. Once predicted, manufacturers take needed steps to prevent this failure '
           'with corrective or scheduled maintenance.')

st.caption('This project in particular will use upwards of 7900 rows of data from boat engine sensors to train '
           'a model that will help predict failures. ')
st.header('')

columns = ["ID", "Date", "Temperature", "Humidity", "Operator", "Measure1", "Measure2", "Measure3", "Measure4",
           "Measure5", "Measure6", "Measure7", "Measure8", "Measure9", "Measure10", "Measure11", "Measure12",
           "Measure13", "Measure14", "Measure15", "Hours Since Previous Failure", "Failure", "Year",
           "Month", "Day of the Month", "Day of the Week", "Hour", "Minute", "Second"]

cols = ["Temperature", "Humidity", "Operator", "Measure1", "Measure2", "Measure3", "Measure4",
        "Measure5", "Measure6", "Measure7", "Measure8", "Measure9", "Measure10", "Measure11", "Measure12",
        "Measure13", "Measure14", "Measure15", "Hours Since Previous Failure", "Failure"]

cols1 = ["Temperature", "Humidity", "Measure1", "Measure2", "Measure3", "Measure4",
        "Measure5", "Measure6", "Measure7", "Measure8", "Measure9", "Measure10", "Measure11", "Measure12",
        "Measure13", "Measure14", "Measure15", "Hours Since Previous Failure"]

train = pd.read_csv("./train.csv", sep=",", names=columns)
test = pd.read_csv("./test.csv", sep=",", names=columns)

train.drop(index=train.index[0], axis=0, inplace=True)

st.header('')
st.header('Part 1: Exploratory data analysis')
st.header('')
st.header('')


st.subheader("Headers")
st.dataframe(train.head())
st.header('')
st.header('')

with st.expander("Unique values"):

    st.subheader("Unique values per column and their counts")
    st.header('')

    for col in cols:
        st.subheader(col)
        st.text(train[col].value_counts())


# st.subheader('Percentage of zero values in rows')
# st.text(train[train == '0'].count(axis=1)/len(train.columns)*100)

st.subheader('Percentage of zero values in columns')
st.text(train[train == '0'].count(axis=0)/len(train.index)*100)

st.text('')
st.text('All of the values in columns Minute and Second are zero, therefore they provide no insight and can be removed')

train = train.drop('Minute', axis=1)
train = train.drop('Second', axis=1)

st.text('Checking for duplicate values (Based on ID):')
st.text(train['ID'].unique().shape[0] != train.shape[0])

train_numeric = train[cols1].copy()
train_numeric = train_numeric.apply(pd.to_numeric)  # convert all columns of DataFrame


st.dataframe(train_numeric.describe())

#
# # Histograms of numeric features
# fig, axs = plt.subplots(nrows=6, ncols=3, figsize=(50, 30))
# fig.suptitle('Numeric features histogram')
# for j, feature in enumerate(cols1):
#     sns.histplot(ax=axs[j//3, j-3*(j//3)], data=train, x=feature)
# st.pyplot(fig)
#
# # boxplot of numeric features
# fig, axs = plt.subplots(nrows=6, ncols=3, figsize=(50, 30))
# fig.suptitle('Numeric features boxplot')
# for j, feature in enumerate(cols1):
#     sns.boxplot(ax=axs[j//3, j-3*(j//3)], data=train, x=feature)
# st.pyplot(fig)

f = train['Failure'].value_counts(normalize=True) * 100

st.subheader("Failures percentage in data:")
st.text(f)

# ###########  UNFINISHED: RESAMPLING FOR MORE BALANCE BETWEEN FAILURES AND NON-FAILURES  #############################
#
# n_working = train['Failure'].value_counts()['No']
# desired_length = round(n_working/0.8)
# spc = round((desired_length-n_working)/4)
#
# balance_cause = {'No': n_working,
#                  'Yes': spc}
# sm = SMOTENC(categorical_features=[0, 7], sampling_strategy=balance_cause, random_state=0)
#
# train_numeric2 = train[cols].copy()
# train_numeric2 = train_numeric2.drop('Operator', axis=1)
#
#
# df_res, y_res = sm.fit_resample(train_numeric2, train_numeric2['Failure'])
# idx_fail_res = df_res.loc[df_res['Failure'] != 'No'].index
# df_res_fail = df_res.loc[idx_fail_res]
#
# st.subheader('Percentage increment of observations after oversampling:')
# st.text(round((df_res.shape[0]-train_numeric2.shape[0])*100/train_numeric2.shape[0], 2))
# st.subheader('SMOTE Resampled Failures percentage:')
# st.text(df_res_fail.shape[0]*100/df_res.shape[0], 2)

#    #############################################################################################################

train_numeric2 = train[cols].copy()
train_numeric2['Failure'] = train_numeric2['Failure'].apply(lambda x: 0 if x == 'No' else 1)
train_numeric2['Operator'] = train_numeric2['Operator'].apply(lambda x: 1 if x == 'Operator1' else 2 if x == 'Operator2'
                                                              else 3 if x == 'Operator3' else 4 if x == 'Operator4'
                                                              else 5 if x == 'Operator5' else 6 if x == 'Operator6'
                                                              else 7 if x == 'Operator7' else 8)

train_numeric2 = train_numeric2.astype('int64')

st.dataframe(train_numeric2.head())

plt.figure(figsize=(40, 40))
sn.heatmap(data=train_numeric2.corr(), mask=np.triu(train_numeric2.corr()), annot=True, cmap='BrBG')
plt.title('Correlation Heatmap')
st.pyplot(plt)


st.text('As we can see, the only two measures with which failures '
        'are relevantly correlated are Temperature and Humidity')


# ===============================================================

scaler = LabelEncoder()
train_numeric2['Failure Type'] = scaler.fit_transform(train_numeric2['Failure Type'])
X = train_numeric2.drop(columns="Failure Type", axis=1)
y = train_numeric2["Failure Type"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)


# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)

log_train = round(logreg.score(X_train, y_train) * 100, 2)
log_accuracy = round(accuracy_score(y_pred_lr, y_test) * 100, 2)


print("Training Accuracy    :", log_train, "%")
print("Model Accuracy Score :", log_accuracy, "%")
print("\033[1m--------------------------------------------------------\033[0m")
print("Classification_Report: \n", classification_report(y_test, y_pred_lr))
print("\033[1m--------------------------------------------------------\033[0m")
plot_confusion_matrix(logreg, X_test, y_test)
plt.title('Confusion Matrix')


# Decision Tree
decision = DecisionTreeClassifier()
decision.fit(X_train, y_train)
y_pred_dec = decision.predict(X_test)

decision_train = round(decision.score(X_train, y_train) * 100, 2)
decision_accuracy = round(accuracy_score(y_pred_dec, y_test) * 100, 2)

print("Training Accuracy    :", decision_train, "%")
print("Model Accuracy Score :", decision_accuracy, "%")
print("\033[1m--------------------------------------------------------\033[0m")
print("Classification_Report: \n", classification_report(y_test, y_pred_dec))
print("\033[1m--------------------------------------------------------\033[0m")
plot_confusion_matrix(decision, X_test, y_test)
plt.title('Confusion Matrix')

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)
random_forest.score(X_train, y_train)

random_forest_train = round(random_forest.score(X_train, y_train) * 100, 2)
random_forest_accuracy = round(accuracy_score(y_pred_rf, y_test) * 100, 2)

print("Training Accuracy    :", random_forest_train, "%")
print("Model Accuracy Score :", random_forest_accuracy, "%")
print("\033[1m--------------------------------------------------------\033[0m")
print("Classification_Report: \n", classification_report(y_test, y_pred_rf))
print("\033[1m--------------------------------------------------------\033[0m")
plot_confusion_matrix(random_forest, X_test, y_test)
plt.title('Confusion Matrix')

# Support Vector Machines
svc = SVC()
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)

svc_train = round(svc.score(X_train, y_train) * 100, 2)
svc_accuracy = round(accuracy_score(y_pred_svc, y_test) * 100, 2)

print("Training Accuracy    :", svc_train, "%")
print("Model Accuracy Score :", svc_accuracy, "%")
print("\033[1m--------------------------------------------------------\033[0m")
print("Classification_Report: \n", classification_report(y_test, y_pred_svc))
print("\033[1m--------------------------------------------------------\033[0m")
plot_confusion_matrix(svc, X_test, y_test)
plt.title('Confusion Matrix')

