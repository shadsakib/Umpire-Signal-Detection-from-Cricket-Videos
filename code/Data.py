import os
import pandas as pd
import numpy as np
import csv
import glob
import re
import csv
from collections import Counter
from sklearn.model_selection import train_test_split

class Data:
    def __init__(self, params):
        self._data_path = os.path.abspath(params['DATA_PATH'])
        self._reqd_labels = params['REQD_LABELS']

        self._label_index_map, self._index_label_map = self._fetch_labels(os.path.abspath(params['LABEL_TEXT']))
        print(self._label_index_map, self._index_label_map)
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.X_test_filenames, self.ordered_test_labels = self._prepare_data()

    def _fetch_labels(self, label_text):
        label_index_map = {}
        index_label_map = {}
        with open(label_text, 'r') as fid:
            csv_fid = csv.reader(fid)
            current_index = 0
            for line_id, label in enumerate(csv_fid):
                if label[0] in self._reqd_labels:
                    label_index_map[label[0]] = current_index
                    index_label_map[current_index] = label[0]
                    current_index += 1

        return label_index_map, index_label_map

    def _prepare_data(self):
        path = '/content/VideoClassification/VideosResized224'

        x, y = [], []
        for name in ['Four', 'NoBall', 'NoSignal', 'Out', 'Six', 'Wide']:
            for video_path in glob.glob(f'{path}/{name}*'):
                filename = os.path.split(video_path)[1]
                x.append(filename)
                y.append(name)

        X = pd.DataFrame()
        X['videoname'] = x
        X['class'] = y

        # train,valid,test split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, stratify=y_train, test_size=0.25, random_state=42)

        c = Counter(y_train)
        d = Counter(y_test)
        e = Counter(y_val)
        
        print(c.most_common(),d.most_common(),e.most_common(), sep='\n')

        train = pd.DataFrame({'videoname':x_train, 'class':y_train})
        test = pd.DataFrame({'videoname':x_test, 'class':y_test})
        validate = pd.DataFrame({'videoname':x_val, 'class':y_val})

        # storing test information
        test_info = [(self.get_numeric(a), b) for a, b in zip(test['videoname'], test['class'])]
        test_info.sort(key=lambda x: x[0])
        files, labels = zip(*test_info)

        X_train, y_train = train['videoname'], train['class']
        X_train = [os.path.join(self._data_path, str(fname)) for fname in X_train]
        y_train = [self._label_index_map[y] for y in y_train]

        X_val, y_val = validate['videoname'], validate['class']
        X_val = [os.path.join(self._data_path, str(fname)) for fname in X_val]
        y_val = [self._label_index_map[y] for y in y_val]

        X_test_filenames, y_test = test['videoname'], test['class']

        #Extract the numeric portion of filenames
        numeric_names = []
        for filename in X_test_filenames:
            match = re.match(r"([a-z]+)([0-9]+)", filename, re.I)
            items = match.groups()
            numeric_names.append(items[1])

        X_test = [os.path.join(self._data_path, str(fname)) for fname in X_test_filenames]

        X_test_filenames = numeric_names.copy()

        print(len(X_train), len(y_train), len(X_val), len(y_val), len(X_test), len(X_test_filenames))

        return X_train, y_train, X_val, y_val, X_test, X_test_filenames, labels


    def label_at_index(self, index):
        return self._index_label_map[index]


    def get_train_data_length(self):
        return len(self.y_train)


    def get_val_data_length(self):
        return len(self.y_val)

    def get_test_data_length(self):
        return len(self.X_test)

    def get_numeric(self, filename):
        match = re.match(r"([a-z]+)([0-9]+)", filename, re.I)
        items = match.groups()
        return int(items[1])



