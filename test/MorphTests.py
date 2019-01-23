import unittest
import pandas as pd
import json
import sys

from elasticsearch import Elasticsearch

sys.path.append(sys.path[0] + "/../")

import dataframe_utils.morphs


def get_test_df():
    # noinspection PyUnresolvedReferences
    index = pd.Series(
        [
            "1542585600000", "1542672000000", "1542758400000", "1542844800000", "1542931200000",
            "1543017600000", "1543104000000", "1543190400000", "1543276800000", "1543363200000"]
    ).astype('int').astype("datetime64[ms]")
    return pd.DataFrame({
        "C1": range(10),
        "C2": ["a"] * 10
    }, index=index)


def get_test_dict():
    return dict({
        '1542585600000': {'C1': 0, 'C2': 'a'},
        '1542672000000': {'C1': 1, 'C2': 'a'},
        '1542758400000': {'C1': 2, 'C2': 'a'},
        '1542844800000': {'C1': 3, 'C2': 'a'},
        '1542931200000': {'C1': 4, 'C2': 'a'},
        '1543017600000': {'C1': 5, 'C2': 'a'},
        '1543104000000': {'C1': 6, 'C2': 'a'},
        '1543190400000': {'C1': 7, 'C2': 'a'},
        '1543276800000': {'C1': 8, 'C2': 'a'},
        '1543363200000': {'C1': 9, 'C2': 'a'}
    })


class MorphTests(unittest.TestCase):

    def test_json_str_to_pandas(self):
        json_str = json.dumps(get_test_dict())
        results = dataframe_utils.morphs.json_str_to_pandas(json_str)
        results.index = results.index.astype('int').astype("datetime64[ms]")

        self.assertTrue(isinstance(results, pd.DataFrame))
        self.assertTrue(results.equals(get_test_df()))

    def test_pandas_to_json_str(self):
        test_df = get_test_df()
        results = dataframe_utils.morphs.pandas_to_json_str(test_df)

        self.assertTrue(isinstance(results, str))
        self.assertDictEqual(json.loads(results), get_test_dict())

    def test_json_dict_to_pandas(self):
        results = dataframe_utils.morphs.json_dict_to_pandas(get_test_dict())
        results.index = results.index.astype('int').astype("datetime64[ms]")

        self.assertTrue(isinstance(results, pd.DataFrame))
        self.assertTrue(results.equals(get_test_df()))

    def test_pandas_to_json_dict(self):
        test_df = get_test_df()
        results = dataframe_utils.morphs.pandas_to_json_dict(test_df)

        self.assertTrue(isinstance(results, dict))
        self.assertDictEqual(results, get_test_dict())

    def test_elasticsearch_connectors(self):
        es = Elasticsearch(["localhost"], scheme="http", port=9200)
        es.indices.delete(index="morphtest", ignore=["404"])

        test_df = get_test_df()
        dataframe_utils.morphs.pandas_to_elasticsearch(test_df, es, "morphtest", chunk_size=3, verbose=False)
        es.indices.flush("morphtest")

        results = dataframe_utils.morphs.elasticsearch_to_pandas(es, "morphtest", chunk_size=3)
        test_df = get_test_df().reset_index(drop=True)
        self.assertTrue(results.reset_index(drop=True).equals(test_df))


if __name__ == '__main__':
    unittest.main(verbosity=2)

