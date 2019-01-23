import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler

from dataframe_utils.dataset.DataframeBinarizer import DataframeBinarizer


class DataLoader:
    """A class for loading and transforming data for the lstm model"""

    def __init__(self):
        self.x_scaler = None
        self.y_scaler = None
        self.x_binarizer = None
        self.y_binarizer = None

        self.split = 0.5
        self.x_data = None
        self.y_data = None

        self.x_cols = None
        self.y_cols = None

    def load_dataset(self, dataframe, split, x_cols, y_cols, cathegorical_cols):
        self.x_cols = x_cols
        self.y_cols = y_cols

        self.split = split

        # build x data
        x_df = dataframe.loc[:, x_cols].reset_index(drop=True)
        self.x_binarizer = DataframeBinarizer()
        x_df = self.x_binarizer.fit_transform(x_df, cathegorical_cols)

        x_data = x_df.values
        self.x_scaler = MinMaxScaler(feature_range=(0, 1))
        self.x_data = self.x_scaler.fit_transform(x_data)

        # build y data
        y_df = dataframe.loc[:, y_cols].reset_index(drop=True)
        self.y_binarizer = DataframeBinarizer()
        y_df = self.y_binarizer.fit_transform(y_df, cathegorical_cols)

        y_data = y_df.values
        self.y_scaler = MinMaxScaler(feature_range=(0, 1))
        self.y_data = self.y_scaler.fit_transform(y_data)

        # if save_dir is not None:
        #    self._save_output_scalers(self.y_scaler, save_dir, model_id)

    def restore_scalers(self, path, model_id):
        x_scaler_filename = path + "/" + model_id + "-x.scaler"
        y_scaler_filename = path + "/" + model_id + "-y.scaler"

        x_binarizer_filename = path + "/" + model_id + "-x.binarizer"
        y_binarizer_filename = path + "/" + model_id + "-y.binarizer"

        self.x_scaler = joblib.load(x_scaler_filename)
        self.y_scaler = joblib.load(y_scaler_filename)
        self.x_binarizer = joblib.load(x_binarizer_filename)
        self.y_binarizer = joblib.load(y_binarizer_filename)

    def save_scalers(self, save_dir, model_id):
        joblib.dump(self.x_scaler, save_dir + "/" + model_id + "-x.scaler")
        joblib.dump(self.y_scaler, save_dir + "/" + model_id + "-y.scaler")

        joblib.dump(self.x_binarizer, save_dir + "/" + model_id + "-x.binarizer")
        joblib.dump(self.y_binarizer, save_dir + "/" + model_id + "-y.binarizer")

    def recompose_results(self, data, *, side):
        assert (side in ["x", "y"])

        scaler = self.x_scaler if side == "x" else self.y_scaler
        inv_yhat = scaler.inverse_transform(data)

        # todo: what about missing and new labels?
        inv_yhat = self.y_binarizer.inverse_transform(inv_yhat)

        return inv_yhat

    def get_test_data(self, in_seq_len, out_seq_len):
        """
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        """
        i_split = int(len(self.x_data) * self.split)
        x_data_test = self.x_data[i_split:]
        y_data_test = self.y_data[i_split:]

        return self._get_data_splits(in_seq_len, out_seq_len, x_data_test, y_data_test)

    def get_train_data(self, in_seq_len, out_seq_len):
        """
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        """
        i_split = int(len(self.x_data) * self.split)
        x_data_train = self.x_data[:i_split]
        y_data_train = self.y_data[:i_split]

        return self._get_data_splits(in_seq_len, out_seq_len, x_data_train, y_data_train)

    def shuffle_data(self):
        stack = np.hstack([self.x_data, self.y_data])
        np.random.shuffle(stack)

        self.x_data = stack[:, :self.x_data.shape[1]]
        self.y_data = stack[:, -self.y_data.shape[1]:]

    @staticmethod
    def _get_sequence_data(data, in_seq_len, out_seq_len):
        data_windows = []
        for i in range(len(data) - (in_seq_len + out_seq_len) + 1):
            data_windows.append(data[i:i + (in_seq_len + out_seq_len)])

        return np.array(data_windows).astype(float)

    @staticmethod
    def _get_data_splits(in_seq_len, out_seq_len, x, y):

        if in_seq_len == 0:
            # don't generate sequences, just format the arrays for consistency
            return x[:, np.newaxis, :], y[:, np.newaxis, :]
        else:
            x = DataLoader._get_sequence_data(x, in_seq_len, out_seq_len)[:, :-out_seq_len, :]
            y = DataLoader._get_sequence_data(y, in_seq_len, out_seq_len)[:, -out_seq_len:, :]
            return x, y

    def parse_x_data(self, x_df):
        x_df = self.x_binarizer.fit_transform(x_df, self.x_binarizer.one_hot_cols)

        x_data = x_df.values
        x_data = self.x_scaler.transform(x_data)

        return x_data
