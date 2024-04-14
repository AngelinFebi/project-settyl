import json
import os.path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

this_dir = os.path.dirname(__file__)


class PRE_PROCESS:
    def __init__(self):
        _file = os.path.join(this_dir, '..', 'src', 'dataset.json')
        with open(_file, 'r') as f:
            _file = json.load(f)
        self.file = _file
        self.df = pd.DataFrame(self.file)
        self.model = self.data_preprocessing()

    def data_preprocessing(self):
        self.df = pd.DataFrame(self.file)
        '''One hot encoding'''
        self.df = pd.get_dummies(self.df, columns=['externalStatus'])
        X = self.df.drop('internalStatus', axis=1)
        y = self.df['internalStatus']
        model = RandomForestClassifier()
        model.fit(X, y)
        return model

    def predict_internal_status(self, external_status):
        df_predict = pd.DataFrame([external_status], columns=['externalStatus'])
        '''Perform one-hot encoding for prediction'''
        df_predict = pd.get_dummies(df_predict, columns=['externalStatus'])
        missing_cols = list(set(self.df.columns) - set(df_predict.columns))
        df_predict = pd.concat([df_predict, pd.DataFrame(0, index=df_predict.index, columns=missing_cols)], axis=1)
        df_predict = df_predict[self.df.drop('internalStatus', axis=1).columns]
        prediction = self.model.predict(df_predict)
        true_labels = {}
        for entry in self.file:
            e_status = entry["externalStatus"]
            i_status = entry["internalStatus"]
            true_labels[e_status] = i_status
        true_label = [true_labels[external_status]]
        predicted_label = [prediction[0]]
        accuracy = accuracy_score(true_label, predicted_label)
        precision = precision_score(true_label, predicted_label, average='weighted')
        recall = recall_score(true_label, predicted_label, average='weighted')
        return prediction[0], accuracy, precision, recall
