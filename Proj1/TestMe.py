import pickle
import pandas as pd
import dataprocessing as dp
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def run_test(model):
    file_name = input("Please enter the name of the file you want to test\n")
    df = dp.clean_data_set(file_name)
    clf = pickle.loads(model)
    # select columns other than 'Persons'
    cols = [col for col in df.columns if col not in ['Persons']]
    # dropping the 'Date' and 'Persons' columns
    data = df[cols]
    # assign the Persons column as target
    target = df['Persons']
    pred = clf.predict(data)
    print("Confusion Matrix", confusion_matrix(pred, target))
    print("Macro-Precision: ", precision_score(pred, target, average='macro'))
    print("Macro-Recall: ", recall_score(pred, target, average='macro'))
    print("Macro-F1: ", f1_score(pred, target, average='macro'))






