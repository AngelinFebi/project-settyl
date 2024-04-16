import json
import os.path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

this_dir = os.path.dirname(__file__)


class PRE_PROCESS:
    def __init__(self, model=None, df=None, _file_=None, model_path=None):
        default_file_path = os.path.join(this_dir, '..', 'src', 'dataset.json')
        if _file_ is None:
            with open(default_file_path, 'r') as f:
                self.file_data = json.load(f)
        else:
            self.file_data = _file_
        self.df = df if df is not None else pd.DataFrame(self.file_data)
        default_model_path = os.path.join(this_dir, '..', 'src', 'model.pkl')
        self.external_status_labels = self.df['externalStatus'].unique().tolist()
        self.model_path = model_path if model_path is not None else default_model_path
        self.model = model if model is not None else self.load_model()

    def data_preprocessing(self):
        self.df = pd.DataFrame(self.df)
        '''One hot encoding'''
        self.df = self.one_hot_encoding(self.df)
        X = self.df.drop('internalStatus', axis=1)
        y = self.df['internalStatus']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        f1score = f1_score(y_test, model.predict(X_test), average='weighted')
        accuracy = accuracy_score(y_test, model.predict(X_test))
        precision = precision_score(y_test, model.predict(X_test), average='weighted')
        recall = recall_score(y_test, model.predict(X_test), average='weighted')
        return model, f1score, accuracy, precision, recall

    def load_model(self):
        if os.path.exists(self.model_path):
            return joblib.load(self.model_path)
        else:
            model, _, _, _, _ = self.data_preprocessing()
            joblib.dump(model, self.model_path)
            return model

    def predict_internal_status(self, external_status):
        df_predict = pd.DataFrame([{"externalStatus": external_status[0]}])
        encoded_column = self.one_hot_encoding(df_predict)
        predicted = self.model.predict(encoded_column)
        _, f1score, accuracy, precision, recall = self.data_preprocessing()
        return predicted[0], round(float(f1score), 4), round(float(accuracy), 4), round(float(precision), 4), round(
            float(recall), 4)

    def one_hot_encoding(self, df):
        df_1_0 = self.one_hot_1_0_map("externalStatus", self.external_status_labels, df)
        df_1_0.drop(columns=['externalStatus'], axis=1, inplace=True)
        return df_1_0

    @staticmethod
    def one_hot_1_0_map(col, labels_list, df):
        new_columns = {}
        for label in labels_list:
            new_columns[label] = np.where(df[col] == label, 1, 0)
        new_columns_df = pd.DataFrame(new_columns, index=df.index)
        df = pd.concat([df, new_columns_df], axis=1)
        return df
