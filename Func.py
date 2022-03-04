import matplotlib.pyplot as plt
import numpy as np
import itertools
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
import seaborn as sns
import numpy as np
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, f1_score, accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt



def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    




def find_cat(x):
    '''x has to be a pd.series'''
    return pd.DataFrame(x.value_counts())


def change_cat(x):
    '''Change to categorical data'''
    ind = find_cat(x).index
    cat_dic = {b:a for a, b in enumerate(ind)}
    return x.replace(cat_dic), cat_dic


def smote_samp(X_train, y_train):
    sm = SMOTE(sampling_strategy='minority', random_state=10)
    oversampled_trainX, oversampled_trainY = sm.fit_sample(X_train, y_train)
    oversampled_train = pd.concat([pd.DataFrame(oversampled_trainY), pd.DataFrame(oversampled_trainX)], axis=1)
    oversampled_train.columns = ['y']+list(X_train.columns)
    print(oversampled_train.shape)
    print(oversampled_train['y'].value_counts())
    return oversampled_train.drop('y',axis=1), oversampled_train['y']


def cv_metric(clf, x,y):
    acc_score = cross_val_score(clf, x, y,cv=10,scoring='accuracy')
    prc = cross_val_score(clf, x, y,cv=10,scoring='precision')
    recall = cross_val_score(clf, x, y,cv=10,scoring='recall')
    f1 = cross_val_score(clf, x, y,cv=10,scoring='f1')
    roc = cross_val_score(clf, x, y,cv=10,scoring='roc_auc')

    print(f'CV accuracy: {np.mean(acc_score):.3f}')
    print(f'CV precision: {np.mean(prc):.3f}')
    print(f'CV recall: {np.mean(recall):.3f}')
    print(f'CV f1: {np.mean(f1):.3f}')
    print(f'CV roc_auc: {np.mean(roc):.3f}')


def model_test(clf, X_train, y_train, X,y):
    classifer = clf()
    classifer.fit(X_train, y_train)
    cv_metric(classifer,X, y)


def cfm(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm =confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index = ['no','yes'],
                     columns = ['no','yes'])
    plt.figure(figsize=(6,6))
    sns.heatmap(df_cm, annot=True)
    plt.title('Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def roc_curve(model,  X_test, y_test):
    probs = model.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

# plot training curve

def train_curve(tup, name):
    df = pd.DataFrame(tup)
    df = df.set_index(0)
    df.plot(kind='line',legend=False, title=f'{name}')

# rank feature importance

def feature_imp(fea_imp, head_dict):
    feature_pair = []
    for a, b in zip(head_dict.values(),fea_imp):
        feature_pair.append((a,b))
    feature_pair = sorted(feature_pair, key= lambda x: x[1], reverse=True)
    return feature_pair