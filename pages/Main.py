import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from mlxtend.plotting import plot_confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE


with st.sidebar:
    # Introductory text
    st.radio('Select one:', [1, 2])
    st.button('Click me')

st.title('Predictive Maintenance with Engine Sensor Data')
st.header('')
st.header('')
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


train = pd.read_csv("train.csv", sep=",", names=columns)
test = pd.read_csv("test.csv", sep=",", names=columns)

train.drop(index=train.index[0], axis=0, inplace=True)

st.header('')
st.header('Exploratory data analysis and data cleaning')
st.header('')


st.subheader('Column headers')
st.dataframe(train.head())
st.header('')
st.header('')

with st.expander('Unique values'):

    st.subheader('Unique values per column and their counts')
    st.header('')

    for col in cols:
        st.subheader(col)
        st.text(train[col].value_counts())

with st.expander('Percentage of zero values in columns'):
    st.text(train[train == '0'].count(axis=0)/len(train.index)*100)
    st.caption('All of the values in columns Minute and Second are zero, therefore '
               'they provide no insight and can be removed')

st.header('')
st.header('')

train = train.drop('Minute', axis=1)
train = train.drop('Second', axis=1)

st.write('Checking for duplicate values (Based on ID):', train['ID'].unique().shape[0] != train.shape[0])

train_numeric = train[cols].copy()
train_numeric['Failure'] = train_numeric['Failure'].apply(lambda x: 0 if x == 'No' else 1)
train_numeric['Operator'] = train_numeric['Operator'].apply(lambda x: 1 if x == 'Operator1' else 2 if x == 'Operator2'
                                                              else 3 if x == 'Operator3' else 4 if x == 'Operator4'
                                                              else 5 if x == 'Operator5' else 6 if x == 'Operator6'
                                                              else 7 if x == 'Operator7' else 8)

train_numeric = train_numeric.astype('int64')

st.header('')
st.header('')
st.subheader('Data after conversion into numeric values')
st.dataframe(train_numeric.head())
st.header('')
st.header('')

st.subheader('Additional information about the data in each column')
st.dataframe(train_numeric.describe().style.background_gradient(cmap="magma"))

st.header('')
st.header('')
st.write('Failure percentage in data')
st.write(train['Failure'].value_counts(normalize=True) * 100)
st.header('')
st.caption('The data appears to be extremely unbalanced in regards to the amount of recorded failures - just 0.9488%.'
           'So it needs to be re-sampled using SMOTE in order to remedy that imbalance.'
           'SMOTE (Synthetic Minority Oversampling Technique) consists of synthesizing elements for the minority class,'
           'based on those that already exist. It works randomly piecing a point from the minority class and computing '
           'the k-nearest neighbors for this point. The synthetic points are added between the chosen point '
           'and its neighbors.')

# Re-sampling data with SMOTE

st.header('')
st.header('')
st.subheader('Before re-sampling')
plt.figure()
sns.scatterplot(data=train_numeric, x='Temperature', y='Humidity', hue='Failure', palette='magma')
st.pyplot(plt)

smote = SMOTE(random_state=0)
X, y = smote.fit_resample(train_numeric[cols], train_numeric['Failure'])
oversampled = pd.DataFrame(X, columns=cols)

st.header('')
st.header('')
st.subheader('After re-sampling')
st.header('')

st.write('Failures percentage in data: ')
st.write(train['Failure'].value_counts(normalize=True) * 100)
plt.figure()
sns.scatterplot(data=oversampled, x='Temperature', y='Humidity', hue='Failure', palette='magma')
st.pyplot(plt)

st.header('')
st.header('')
st.subheader('Re-sampled data info: ')
st.write(oversampled.describe().style.background_gradient(cmap="magma"))

st.header('')
st.header('')
st.subheader('Correlation Heatmap')
plt.figure(figsize=(50, 50))
sns.heatmap(data=oversampled.corr(), annot=True, cmap='magma')
plt.style.use('dark_background')
st.pyplot(plt)


st.caption('As we can see, the only two measures with which failures'
           'are relevantly correlated are Temperature and Humidity')

scaler = LabelEncoder()
oversampled['Failure'] = scaler.fit_transform(oversampled['Failure'])
X = oversampled.drop(columns='Failure', axis=1)
y = oversampled['Failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

st.write("--------------------------------------------------------")

st.header('Training models and comparing results')
st.header('')
# Logistic Regression
logreg = LogisticRegression(random_state=0)
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)

log_train = round(logreg.score(X_train, y_train) * 100, 2)
log_accuracy = round(accuracy_score(y_pred_lr, y_test) * 100, 2)

st.subheader('Logistic Regression')
st.write('Training Accuracy    :', log_train, '%')
st.write('Model Accuracy Score :', log_accuracy, '%')
s = 'Classification_Report: \n' + classification_report(y_test, y_pred_lr)
st.text(s)
cm = confusion_matrix(y_test, y_pred_lr, labels=logreg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=logreg.classes_)
disp.plot(cmap='magma')
st.pyplot(plt)
st.write('--------------------------------------------------------')


# Decision Tree
decision = DecisionTreeClassifier(random_state=0)
decision.fit(X_train, y_train)
y_pred_dec = decision.predict(X_test)

decision_train = round(decision.score(X_train, y_train) * 100, 2)
decision_accuracy = round(accuracy_score(y_pred_dec, y_test) * 100, 2)

st.subheader('Decision Tree Classifier')
st.write('Training Accuracy    :', decision_train, '%')
st.write('Model Accuracy Score :', decision_accuracy, '%')
s = 'Classification_Report: \n' + classification_report(y_test, y_pred_dec)
st.text(s)
cm = confusion_matrix(y_test, y_pred_dec, labels=decision.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=decision.classes_)
disp.plot(cmap='magma')
st.pyplot(plt)
st.write('--------------------------------------------------------')


# Random Forest
random_forest = RandomForestClassifier(n_estimators=100, random_state=0)
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)
random_forest.score(X_train, y_train)

random_forest_train = round(random_forest.score(X_train, y_train) * 100, 2)
random_forest_accuracy = round(accuracy_score(y_pred_rf, y_test) * 100, 2)

st.subheader('Random Forest Classifier')
st.write('Training Accuracy    :', random_forest_train, '%')
st.write('Model Accuracy Score :', random_forest_accuracy, '%')
s = 'Classification_Report: \n' + classification_report(y_test, y_pred_rf)
st.text(s)
cm = confusion_matrix(y_test, y_pred_rf, labels=random_forest.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=random_forest.classes_)
disp.plot(cmap='magma')
st.pyplot(plt)
st.write('--------------------------------------------------------')

# Support Vector Machines
svc = SVC(random_state=0)
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)

svc_train = round(svc.score(X_train, y_train) * 100, 2)
svc_accuracy = round(accuracy_score(y_pred_svc, y_test) * 100, 2)

st.subheader('Support Vector Machines')
st.write('Training Accuracy    :', svc_train, '%')
st.write('Model Accuracy Score :', svc_accuracy, '%')
s = 'Classification_Report: \n' + classification_report(y_test, y_pred_svc)
st.text(s)
cm = confusion_matrix(y_test, y_pred_svc, labels=svc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=svc.classes_)
disp.plot(cmap='magma')
st.pyplot(plt)
st.write('--------------------------------------------------------')


models = pd.DataFrame({
    'Model': [
        'Logistic Regression', 'Decision Tree',  'Random Forest', 'Support Vector Machines',
    ],

    'Training Accuracy':
        [log_train, decision_train, random_forest_train, svc_train],

    'Model Accuracy Score': [
        log_accuracy, decision_accuracy, random_forest_accuracy,  svc_accuracy ]
})

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
st.subheader('Choosing Random Forest Classifier because it has the highest model accuracy score of 99.94%')

prediction1 = decision.predict(X_test)

cross_checking = pd.DataFrame({'Actual': y_test, 'Predicted': prediction1})
cross_checking.sample(5).style.background_gradient(
        cmap='coolwarm').set_properties(**{
            'font-family': 'Lucida Calligraphy',
            'color': 'LigntGreen',
            'font-size': '15px'
        })

st.write(cross_checking.style.background_gradient(cmap='Purples'))
