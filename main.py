import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, fbeta_score
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import time
from sklearn import tree
import graphviz

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

train = pd.read_csv("/Users/Luda/PycharmProjects/predictiveMaintenance/input/train.csv", sep=",", names=columns)
test = pd.read_csv("/Users/Luda/PycharmProjects/predictiveMaintenance/input/test.csv", sep=",", names=columns)
# test_results = pd.read_csv("/Users/Luda/PycharmProjects/predictiveMaintenance/input/RUL_FD001.txt", sep=" ", header=None)

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

X, y = train_numeric2[cols], train_numeric2[['Failure']]
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.1, stratify=train_numeric2['Failure'],
                                                          random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.11,
                                                  stratify=y_trainval['Failure'], random_state=0)


def eval_preds(model,X,y_true,y_pred,task):
    if task == 'binary':
        # Extract task target
        y_true = y_true['Failure']
        cm = confusion_matrix(y_true, y_pred)
        # Probability of the minority class
        proba = model.predict_proba(X)[:, 1]
        # Metrics
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, proba)
        f1 = f1_score(y_true, y_pred, pos_label=1)
        f2 = fbeta_score(y_true, y_pred, pos_label=1, beta=2)

        metrics = pd.Series(data={'ACC': acc, 'AUC': auc, 'F1': f1, 'F2': f2})
        metrics = round(metrics, 3)
        return cm, metrics


def tune_and_fit(clf, X, y, params, task):
    if task == 'binary':
        f2_scorer = make_scorer(fbeta_score, pos_label=1, beta=2)
        start_time = time.time()
        grid_model = GridSearchCV(clf, param_grid=params,
                                  cv=5, scoring=f2_scorer)
        grid_model.fit(X, y['Failure'])


    s = 'Best params:'+grid_model.best_params_
    st.text(s)
    train_time = time.time()-start_time
    mins = int(train_time//60)
    s1 = 'Training time: '+str(mins)+'m '+str(round(train_time-mins*60))+'s'
    st.text(s1)
    return grid_model


def predict_and_evaluate(fitted_models, X, y_true, clf_str, task):
    cm_dict = {key: np.nan for key in clf_str}
    metrics = pd.DataFrame(columns=clf_str)
    y_pred = pd.DataFrame(columns=clf_str)
    for fit_model, model_name in zip(fitted_models, clf_str):
        # Update predictions
        y_pred[model_name] = fit_model.predict(X)
        # Metrics
        if task == 'binary':
            cm, scores = eval_preds(fit_model, X, y_true, y_pred[model_name], task)
        elif task == 'multi_class':
            cm, scores = eval_preds(fit_model, X, y_true, y_pred[model_name], task)
        # Update Confusion matrix and metrics
        cm_dict[model_name] = cm
        metrics[model_name] = scores
    return y_pred, cm_dict, metrics


def fit_models(clf, clf_str, X_train, X_val, y_train, y_val):
    metrics = pd.DataFrame(columns=clf_str)
    for model, model_name in zip(clf, clf_str):
        model.fit(X_train, y_train['Failure'])
        y_val_pred = model.predict(X_val)
        metrics[model_name] = eval_preds(model, X_val, y_val, y_val_pred, 'binary')[1]
    return metrics


lr = LogisticRegression()
knn = KNeighborsClassifier()
svc = SVC(probability=True)
rfc = RandomForestClassifier()
xgb = XGBClassifier()

clf = [lr, knn, svc, rfc, xgb]
clf_str = ['LR', 'KNN', 'SVC', 'RFC', 'XGB']

# Fit on raw train
metrics_0 = fit_models(clf, clf_str, X_train, X_val, y_train, y_val)

# Fit on temperature product train
XX_train = X_train.drop(columns=['Temperature'])
XX_val = X_val.drop(columns=['Temperature'])
XX_train['Temperature'] = X_train['Temperature']
XX_val['Temperature'] = X_val['Temperature']
metrics_1 = fit_models(clf, clf_str, XX_train, XX_val, y_train, y_val)

# Fit on power product train
XX_train = X_train.drop(columns=['Humidity'])
XX_val = X_val.drop(columns=['Humidity'])
XX_train['Humidity'] = X_train['Humidity']
XX_val['Humidity'] = X_val['Humidity']
metrics_2 = fit_models(clf, clf_str, XX_train, XX_val, y_train, y_val)

# Fit on both products train
XX_train = X_train.drop(columns=['Temperature', 'Humidity'])
XX_val = X_val.drop(columns=['Temperature', 'Humidity'])
XX_train['Temperature']= X_train['Temperature']
XX_val['Temperature']= X_val['Temperature']
XX_train['Humidity'] = X_train['Humidity']
XX_val['Humidity'] = X_val['Humidity']
metrics_3 = fit_models(clf, clf_str, XX_train, XX_val, y_train, y_val)

# classification metrics barplot
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 8))
fig.suptitle('Classification metrics')
for j, model in enumerate(clf_str):
    ax = axs[j//3, j-3*(j//3)]
    model_metrics = pd.DataFrame(data=[metrics_0[model], metrics_1[model], metrics_2[model], metrics_3[model]])
    model_metrics.index = ['Original', 'Temperature', 'Humidity', 'Both']
    model_metrics.transpose().plot(ax=ax, kind='bar', rot=0, )
    ax.title.set_text(model)
    ax.get_legend().remove()
fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)
axs.flatten()[-2].legend(title='Dataset', loc='upper center',
                         bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=12)
st.pyplot(plt)

# Make predictions
lr = LogisticRegression(random_state=0)
lr.fit(X_train, y_train['Failure'])
y_val_lr = lr.predict(X_val)
y_test_lr = lr.predict(X_test)

# Metrics
cm_val_lr, metrics_val_lr = eval_preds(lr,X_val,y_val,y_val_lr,'binary')
cm_test_lr, metrics_test_lr = eval_preds(lr,X_test,y_test,y_test_lr,'binary')
# s = 'Validation set metrics:' + metrics_val_lr + '\n'
# st.subheader(s)
# s1 = 'Test set metrics:' + metrics_test_lr + '\n'
# st.subheader(s1)


cm_labels = ['Not Failure', 'Failure']
cm_lr = [cm_val_lr, cm_test_lr]
# Show Confusion Matrices
fig, axs = plt.subplots(ncols=2, figsize=(8,4))
fig.suptitle('LR Confusion Matrices')
for j, title in enumerate(['Validation Set', 'Test Set']):
    ax = axs[j]
    sn.heatmap(ax=ax, data=cm_lr[j], annot=True,
              fmt='d', cmap='Blues', cbar=False)
    axs[j].title.set_text(title)
    axs[j].set_xticklabels(cm_labels)
    axs[j].set_yticklabels(cm_labels)
st.pyplot(plt)

# Odds for interpretation
d = {'feature': X_train.columns, 'odds': np.exp(lr.coef_[0])}
odds_df = pd.DataFrame(data=d).sort_values(by='odds', ascending=False)
odds_df


knn = KNeighborsClassifier()
svc = SVC()
rfc = RandomForestClassifier()
xgb = XGBClassifier()
clf = [knn, svc, rfc, xgb]
clf_str = ['KNN', 'SVC', 'RFC', 'XGB']

# Parameter grids for GridSearch
knn_params = {'n_neighbors':[1,3,5,8,10]}
svc_params = {'C': [1, 10, 100],
              'gamma': [0.1,1],
              'kernel': ['rbf'],
              'probability':[True],
              'random_state':[0]}
rfc_params = {'n_estimators':[100,300,500,700],
              'max_depth':[5,7,10],
              'random_state':[0]}
xgb_params = {'n_estimators':[300,500,700],
              'max_depth':[5,7],
              'learning_rate':[0.01,0.1],
              'objective':['binary:logistic']}
params = pd.Series(data=[knn_params,svc_params,rfc_params,xgb_params],
                   index=clf)


##################################### GRID SEARCH - UNFINISHED ##################################################

# st.subheader('GridSearch start')
# fitted_models_binary = []

# for model, model_name in zip(clf, clf_str):
#     print('Training '+str(model_name))
#     fit_model = tune_and_fit(model, X_train, y_train, params[model], 'binary')
#     fitted_models_binary.append(fit_model)
#
# # Create evaluation metrics
# task = 'binary'
# y_pred_val, cm_dict_val, metrics_val = predict_and_evaluate(
#     fitted_models_binary, X_val, y_val, clf_str, task)
# y_pred_test, cm_dict_test, metrics_test = predict_and_evaluate(
#     fitted_models_binary, X_test, y_test, clf_str, task)
#
# # Show Validation Confusion Matrices
# fig, axs = plt.subplots(ncols=4, figsize=(20, 4))
# fig.suptitle('Validation Set Confusion Matrices')
# for j, model_name in enumerate(clf_str):
#     ax = axs[j]
#     sn.heatmap(ax=ax, data=cm_dict_val[model_name], annot=True,
#                 fmt='d', cmap='Blues', cbar=False)
#     ax.title.set_text(model_name)
#     ax.set_xticklabels(cm_labels)
#     ax.set_yticklabels(cm_labels)
# st.pyplot(plt)
#
# # Show Test Confusion Matrices
# fig, axs = plt.subplots(ncols=4, figsize=(20, 4))
# fig.suptitle('Test Set Confusion Matrices')
# for j, model_name in enumerate(clf_str):
#     ax = axs[j]
#     sn.heatmap(ax=ax, data=cm_dict_test[model_name], annot=True,
#                 fmt='d', cmap='Blues', cbar=False)
#     ax.title.set_text(model_name)
#     ax.set_xticklabels(cm_labels)
#     ax.set_yticklabels(cm_labels)
# st.pyplot(plt)
#
# # Print scores
# print('')
# print('Validation scores:', metrics_val, sep='\n')
# print('Test scores:', metrics_test, sep='\n')


# # Evaluate Permutation Feature Importances
# f2_scorer = make_scorer(fbeta_score, pos_label=1, beta=2)
# importances = pd.DataFrame()
# for clf in fitted_models_binary:
#     result = permutation_importance(clf, X_train,y_train['Target'],
#                                   scoring=f2_scorer,random_state=0)
#     result_mean = pd.Series(data=result.importances_mean, index=X.columns)
#     importances = pd.concat(objs=[importances,result_mean],axis=1)
# importances.columns = clf_str
#
# # Barplot of Feature Importances
# fig, axs = plt.subplots(ncols=4, figsize=(20,4))
# fig.suptitle('Permutation Feature Importances')
# for j, name in enumerate(importances.columns):
#     sns.barplot(ax=axs[j], x=importances.index, y=importances[name].values)
#     axs[j].tick_params('x',labelrotation=90)
#     axs[j].set_ylabel('Importances')
#     axs[j].title.set_text(str(name))
# st.pyplot(plt)
###############################################################################################################

# multiclass classification
lr = LogisticRegression(random_state=0, multi_class='ovr')
lr.fit(X_train, y_train['Failure'])
y_val_lr = lr.predict(X_val)
y_test_lr = lr.predict(X_test)
#
# # Validation metrics
# cm_val_lr, metrics_val_lr = eval_preds(lr, X_val, y_val, y_val_lr, 'multi_class')
# cm_test_lr, metrics_test_lr = eval_preds(lr, X_test, y_test, y_test_lr, 'multi_class')
# s = 'Validation set metrics:' + metrics_val_lr + '\n'
# st.text(s)
# s1 = 'Test set metrics:' + metrics_test_lr + '\n'
# st.text(s1)
#
# cm_lr = [cm_val_lr, cm_test_lr]
# cm_labels = ['No Failure', 'Failure']
# # Show Confusion Matrices
# fig, axs = plt.subplots(ncols=2, figsize=(9, 4))
# fig.suptitle('LR Confusion Matrices')
# for j, title in enumerate(['Validation Set', 'Test Set']):
#     ax = axs[j]
#     sn.heatmap(ax=ax, data=cm_lr[j], annot=True,
#                fmt='d', cmap='Blues', cbar=False)
#     axs[j].title.set_text(title)
#     axs[j].set_xticklabels(cm_labels)
#     axs[j].set_yticklabels(cm_labels)
# st.pyplot(plt)

# Odds for interpretation
# odds_df = pd.DataFrame(data=np.exp(lr.coef_), columns=X_train.columns,
#                        index=train_numeric2['Failure'].unique())
# odds_df
#
# # Models
# knn = KNeighborsClassifier()
# svc = SVC(decision_function_shape='ovr')
# rfc = RandomForestClassifier()
# xgb = XGBClassifier()
# clf = [knn, svc, rfc, xgb]
# clf_str = ['KNN', 'SVC', 'RFC', 'XGB']
#
# knn_params = {'n_neighbors':[1, 3, 5, 8, 10]}
# svc_params = {'C': [1, 10, 100],
#               'gamma': [0.1, 1],
#               'kernel': ['rbf'],
#               'probability':[True],
#               'random_state':[0]}
# rfc_params = {'n_estimators':[100, 300, 500, 700],
#               'max_depth':[5, 7, 10],
#               'random_state':[0]}
# xgb_params = {'n_estimators':[100, 300, 500],
#               'max_depth':[5, 7, 10],
#               'learning_rate':[0.01, 0.1],
#               'objective':['multi:softprob']}
#
# params = pd.Series(data=[knn_params, svc_params, rfc_params, xgb_params],
#                    index=clf)
#
#
# # Tune hyperparameters with GridSearch (estimated time 8-10m)
# print('GridSearch start')
# fitted_models_multi = []
# for model, model_name in zip(clf, clf_str):
#     print('Training '+str(model_name))
#     fit_model = tune_and_fit(model, X_train, y_train, params[model], 'multi_class')
#     fitted_models_multi.append(fit_model)
#
#
# # Create evaluation metrics
#
# task = 'multi_class'
# y_pred_val, cm_dict_val, metrics_val = predict_and_evaluate(
#     fitted_models_multi, X_val, y_val, clf_str, task)
# y_pred_test, cm_dict_test, metrics_test = predict_and_evaluate(
#     fitted_models_multi, X_test, y_test, clf_str, task)
#
# # Show Validation Confusion Matrices
# fig, axs = plt.subplots(ncols=4, figsize=(20, 4))
# fig.suptitle('Validation Set Confusion Matrices')
# for j, model_name in enumerate(clf_str):
#     ax = axs[j]
#     sn.heatmap(ax=ax, data=cm_dict_val[model_name], annot=True,
#                fmt='d', cmap='Blues', cbar=False)
#     ax.title.set_text(model_name)
#     ax.set_xticklabels(cm_labels)
#     ax.set_yticklabels(cm_labels)
# st.pyplot(plt)
#
# # Show Test Confusion Matrices
# fig, axs = plt.subplots(ncols=4, figsize=(20,4))
# fig.suptitle('Test Set Confusion Matrices')
# for j, model_name in enumerate(clf_str):
#     ax = axs[j]
#     sn.heatmap(ax=ax, data=cm_dict_test[model_name], annot=True,
#                fmt='d', cmap='Blues', cbar=False)
#     ax.title.set_text(model_name)
#     ax.set_xticklabels(cm_labels)
#     ax.set_yticklabels(cm_labels)
# st.pyplot(plt)
#
# # Print scores
# print('')
# print('Validation scores:', metrics_val, sep='\n')
# print('Test scores:', metrics_test, sep='\n')
#
#
# # Evaluate Permutation Feature Importances
# f2_scorer = make_scorer(fbeta_score, beta=2, average='weighted')
# importances = pd.DataFrame()
# for clf in fitted_models_multi:
#     result = permutation_importance(clf, X_train, y_train['Failure Type'],
#                                   scoring=f2_scorer, random_state=0)
#     result_mean = pd.Series(data=result.importances_mean, index=X.columns)
#     importances = pd.concat(objs=[importances, result_mean], axis=1)
#
# importances.columns = clf_str
#
# # Barplot of Feature Importances
# fig, axs = plt.subplots(ncols=4, figsize=(20,4))
# fig.suptitle('Permutation Feature Importances')
# for j, name in enumerate(importances.columns):
#     sn.barplot(ax=axs[j], x=importances.index, y=importances[name].values)
#     axs[j].tick_params('x', labelrotation=90)
#     axs[j].set_ylabel('Importances')
#     axs[j].title.set_text(str(name))
# st.pyplot(plt)
# # Random Forest Decision Path
#
#
# tree_binary = fitted_models_binary[2].best_estimator_.estimators_[0]
# tree_multi = fitted_models_multi[2].best_estimator_.estimators_[0]
# trees = [tree_binary,tree_multi]
# targets = ['Failure']
# for decision_tree, target in zip(trees, targets):
#     decision_tree.fit(X_train, y_train[target])
#     classes = list(map(str, train_numeric2[target].unique()))
#
#     dot_data = tree.export_graphviz(decision_tree, out_file=None,
#                                   feature_names=X.columns,
#                                   class_names=classes,
#                                   filled=True, rounded=True,
#                                   special_characters=True,
#                                   max_depth=4)  # uncomment to see full tree
#     graph = graphviz.Source(dot_data)
#     graph.render(target+" Classification tree")
#     display(graph)
