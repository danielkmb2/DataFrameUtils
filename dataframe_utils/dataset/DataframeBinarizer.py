import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class DataframeBinarizer:
    def __init__(self):
        self.label_encoders = {}
        self.onehot_encoders = {}
        self.one_hot_cols = []
        self.y_cols = []

    def fit_transform(self, _df, one_hot_cols):
        self.one_hot_cols = list(set(one_hot_cols).intersection(set(_df.columns)))

        for col in self.one_hot_cols:
            self.label_encoders[col] = LabelEncoder()
            self.onehot_encoders[col] = OneHotEncoder(categories="auto")

            _df[col] = self.label_encoders[col].fit_transform(_df[col])
            df_one_hot = self.onehot_encoders[col].fit_transform(_df[col].values.reshape(-1, 1)).toarray()

            onehot_cols = ["_" + col + "_" + str(int(i)) for i in range(df_one_hot.shape[1])]
            df_one_hot = pd.DataFrame(df_one_hot, columns=onehot_cols)
            _df = pd.concat([_df, df_one_hot], axis=1)
            _df.drop([col], axis=1, inplace=True)

        self.y_cols = _df.columns.tolist()
        return _df

    def transform(self, _df):
        for col in self.one_hot_cols:
            _df[col] = self.label_encoders[col].transform(_df[col])
            df_one_hot = self.onehot_encoders[col].transform(_df[col].values.reshape(-1, 1)).toarray()

            onehot_cols = ["_" + col + "_" + str(int(i)) for i in range(df_one_hot.shape[1])]
            df_one_hot = pd.DataFrame(df_one_hot, columns=onehot_cols)
            _df = pd.concat([_df, df_one_hot], axis=1)
            _df.drop([col], axis=1, inplace=True)

        return _df

    def inverse_transform(self, _df):
        _df = pd.DataFrame(data=_df, columns=self.y_cols)

        for col in self.one_hot_cols:
            onehot_cols = _df.columns[pd.Series(_df.columns).str.startswith("_" + col + "_")]

            df_one_hot = _df[onehot_cols]
            df_one_hot = self.onehot_encoders[col].inverse_transform(df_one_hot)
            df_one_hot = self.label_encoders[col].inverse_transform(df_one_hot.round().ravel().astype(int))

            _df.drop(onehot_cols, axis=1, inplace=True)
            _df[col] = df_one_hot

        return _df
