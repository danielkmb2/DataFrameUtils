from typing import List

from dataframe_utils.dataset.DataLoader import DataLoader


def _resume_splits(splits, x_cols, y_cols, dataloader):
    (x_train, y_train, x_test, y_test) = splits

    def resume_outputs(cols, binarizer):
        print(" - Original columns: " + str(cols))
        print(" - Binarezed columns: " + str(binarizer.one_hot_cols))
        print(" - Output columns: " + str(binarizer.y_cols))

    print("X_DATA:")
    resume_outputs(x_cols, dataloader.x_binarizer)
    print(" - Output X train shape: " + str(x_train.shape))
    print(" - Output X test shape: " + str(x_test.shape))

    print()

    print("Y_DATA:")
    resume_outputs(y_cols, dataloader.y_binarizer)
    print(" - Output Y train shape: " + str(y_train.shape))
    print(" - Output Y test shape: " + str(y_test.shape))


def get_sampled_splits(df, split=0.8, x_cols=None, y_cols=None, cathegorical_cols=None, shuffle=False, print_info=True):

    if cathegorical_cols is None:
        cathegorical_cols = []
    if y_cols is None:
        y_cols = []
    if x_cols is None:
        x_cols = []

    assert isinstance(cathegorical_cols, List)
    assert isinstance(x_cols, List)
    assert isinstance(y_cols, List)
    assert len(x_cols) > 0
    assert len(y_cols) > 0

    dataloader = DataLoader()
    dataloader.load_dataset(df, split, x_cols, y_cols, cathegorical_cols)
    if shuffle:
        dataloader.shuffle_data()

    x_train, y_train = dataloader.get_train_data(in_seq_len=0, out_seq_len=0)
    x_test, y_test = dataloader.get_test_data(in_seq_len=0, out_seq_len=0)

    (x_train, y_train, x_test, y_test) = (s[:, 0, :] for s in (x_train, y_train, x_test, y_test))

    if print_info:
        _resume_splits((x_train, y_train, x_test, y_test), x_cols, y_cols, dataloader)
    return (x_train, y_train, x_test, y_test), dataloader


def get_sequenced_splits(df, in_seq_len=2, out_seq_len=1, split=0.8, x_cols=None, y_cols=None, cathegorical_cols=None,
                         print_info=True):

    if cathegorical_cols is None:
        cathegorical_cols = []
    if y_cols is None:
        y_cols = []
    if x_cols is None:
        x_cols = []

    assert isinstance(cathegorical_cols, List)
    assert isinstance(x_cols, List)
    assert isinstance(y_cols, List)
    assert len(x_cols) > 0
    assert len(y_cols) > 0

    assert in_seq_len > 0
    assert out_seq_len > 0
    assert (in_seq_len + out_seq_len) <= (df.shape[0] * split)
    assert (in_seq_len + out_seq_len) <= (df.shape[0] * (1 - split))

    dataloader = DataLoader()
    dataloader.load_dataset(df, split, x_cols, y_cols, cathegorical_cols)

    x_train, y_train = dataloader.get_train_data(in_seq_len=in_seq_len, out_seq_len=out_seq_len)
    x_test, y_test = dataloader.get_test_data(in_seq_len=in_seq_len, out_seq_len=out_seq_len)

    if print_info:
        _resume_splits((x_train, y_train, x_test, y_test), x_cols, y_cols, dataloader)
    return (x_train, y_train, x_test, y_test), dataloader


def recompose_x_data(results, dataloader):
    inv_x_data = dataloader.recompose_results(results, side="x")
    inv_x_data.columns = dataloader.x_cols
    return inv_x_data


def recompose_y_data(results, dataloader):
    return dataloader.recompose_results(results, side="y")


def save_dataloader(dataloader, path, model_id):
    dataloader.save_scalers(path, model_id)


def restore_dataloader(path, model_id):
    dataloader = DataLoader()
    dataloader.restore_scalers(path, model_id)

    return dataloader


def prepare_sampled_x_data(dataloader, new_samples):
    return dataloader.parse_x_data(new_samples)


def prepare_sequenced_x_data(dataloader, new_samples, in_seq_len=1):
    scaled_samples = dataloader.parse_x_data(new_samples)
    return DataLoader._get_sequence_data(scaled_samples, in_seq_len, 0)
