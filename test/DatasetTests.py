import unittest
import pandas as pd
import numpy as np
import sys

sys.path.append(sys.path[0] + "/../")

import dataframe_utils.dataset


def get_test_df():
    return pd.DataFrame({
        "C1": range(20),
        "C2": ["a", "a", "b", "a", "a", "a", "a", "a", "b", "b", "a", "a", "b", "a", "a", "a", "a", "a", "b", "b"],
        "C3": range(20),
        "C4": ["a", "a", "b", "a", "a", "a", "a", "a", "b", "b", "a", "a", "b", "a", "a", "a", "a", "a", "b", "b"]
    })


class DatasetTests(unittest.TestCase):
    def test_sequence_builder(self):
        df = get_test_df()
        splits, dataloader = dataframe_utils.dataset.get_sequenced_splits(df, in_seq_len=1, out_seq_len=1, split=0.5,
                                                                          x_cols=["C1", "C2"], y_cols=["C3", "C4"],
                                                                          cathegorical_cols=["C2", "C4"],
                                                                          print_info=False)
        x_train, y_train, x_test, y_test = splits

        original_x_train = dataframe_utils.dataset.recompose_x_data(x_train[:, 0, :], dataloader)
        original_y_train = dataframe_utils.dataset.recompose_y_data(y_train[:, 0, :], dataloader)
        original_x_test = dataframe_utils.dataset.recompose_x_data(x_test[:, 0, :], dataloader)
        original_y_test = dataframe_utils.dataset.recompose_y_data(y_test[:, 0, :], dataloader)

        original = pd.concat([original_x_train, original_y_train], axis=1)
        original.C1 = original.C1.round(1)
        original.C3 = original.C3.round(1)

        expected = pd.DataFrame({
            "C1": range(9),
            "C2": ["a", "a", "b", "a", "a", "a", "a", "a", "b"],
            "C3": range(1, 10),
            "C4": ["a", "b", "a", "a", "a", "a", "a", "b", "b"]
        })

        self.assertTrue(np.array_equal(original.values, expected.values))

        original = pd.concat([original_x_test, original_y_test], axis=1)
        expected = pd.DataFrame({
            "C1": range(10, 19),
            "C2": ["a", "a", "b", "a", "a", "a", "a", "a", "b"],
            "C3": range(11, 20),
            "C4": ["a", "b", "a", "a", "a", "a", "a", "b", "b"]
        })
        original.C1 = original.C1.round(1)
        original.C3 = original.C3.round(1)
        self.assertTrue(np.array_equal(original.values, expected.values))

    def test_sampled_builder(self):
        df = get_test_df()
        splits, dataloader = dataframe_utils.dataset.get_sampled_splits(df, split=0.5,
                                                                        x_cols=["C1", "C2"], y_cols=["C3", "C4"],
                                                                        cathegorical_cols=["C2", "C4"],
                                                                        print_info=False)
        x_train, y_train, x_test, y_test = splits

        original_x_train = dataframe_utils.dataset.recompose_x_data(x_train, dataloader)
        original_y_train = dataframe_utils.dataset.recompose_y_data(y_train, dataloader)
        original_x_test = dataframe_utils.dataset.recompose_x_data(x_test, dataloader)
        original_y_test = dataframe_utils.dataset.recompose_y_data(y_test, dataloader)

        original = pd.concat([original_x_train, original_y_train], axis=1)
        original.C1 = original.C1.round(1)
        original.C3 = original.C3.round(1)

        expected = pd.DataFrame({
            "C1": range(10),
            "C2": ["a", "a", "b", "a", "a", "a", "a", "a", "b", "b"],
            "C3": range(10),
            "C4": ["a", "a", "b", "a", "a", "a", "a", "a", "b", "b"]
        })

        self.assertTrue(np.array_equal(original.values, expected.values))

        original = pd.concat([original_x_test, original_y_test], axis=1)
        original.C1 = original.C1.round(1)
        original.C3 = original.C3.round(1)

        expected = pd.DataFrame({
            "C1": range(10, 20),
            "C2": ["a", "a", "b", "a", "a", "a", "a", "a", "b", "b"],
            "C3": range(10, 20),
            "C4": ["a", "a", "b", "a", "a", "a", "a", "a", "b", "b"]
        })

        self.assertTrue(np.array_equal(original.values, expected.values))

    @staticmethod
    def _delete_model_files():
        import os

        files = ["test-x.binarizer", "test-y.binarizer", "test-x.scaler", "test-y.scaler"]
        for file in files:
            os.remove(file)

    def test_sampled_scaler_reloading(self):
        df = get_test_df()
        splits, dataloader = dataframe_utils.dataset.get_sampled_splits(df, split=0.5,
                                                                        x_cols=["C1", "C2"], y_cols=["C3", "C4"],
                                                                        cathegorical_cols=["C2", "C4"],
                                                                        print_info=False)
        x_train, y_train, x_test, y_test = splits

        dataframe_utils.dataset.save_dataloader(dataloader, ".", "test")
        new_dataloader = dataframe_utils.dataset.restore_dataloader(".", "test")

        new_x_test = df.loc[10:, ["C1", "C2"]].reset_index(drop=True)
        new_x_test = dataframe_utils.dataset.prepare_sampled_x_data(new_dataloader, new_x_test)

        self.assertTrue(np.array_equal(x_test, new_x_test))
        self._delete_model_files()

    def test_sequenced_scaler_reloading(self):
        df = get_test_df()
        splits, dataloader = dataframe_utils.dataset.get_sequenced_splits(df, in_seq_len=3, out_seq_len=1, split=0.6,
                                                                          x_cols=["C1", "C2"], y_cols=["C3", "C4"],
                                                                          cathegorical_cols=["C2", "C4"],
                                                                          print_info=False)
        x_train, y_train, x_test, y_test = splits

        dataframe_utils.dataset.save_dataloader(dataloader, ".", "test")
        new_dataloader = dataframe_utils.dataset.restore_dataloader(".", "test")

        new_x_test = df.loc[12:18, ["C1", "C2"]].reset_index(drop=True)
        new_x_test = dataframe_utils.dataset.prepare_sequenced_x_data(new_dataloader, new_x_test, in_seq_len=3)

        self.assertTrue(np.array_equal(x_test, new_x_test))
        self._delete_model_files()


if __name__ == '__main__':
    unittest.main(verbosity=2)


