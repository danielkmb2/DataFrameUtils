import math

import pandas as pd
import json
from elasticsearch import Elasticsearch
from progress.bar import Bar  # sudo pip install progress


def json_str_to_pandas(json_str: str) -> pd.DataFrame:
    json_dict = json.loads(json_str)
    return pd.DataFrame.from_dict(json_dict, orient="index")


def pandas_to_json_str(df: pd.DataFrame) -> str:
    return df.to_json(orient='index')


def json_dict_to_pandas(json_dict: dict) -> pd.DataFrame:
    return pd.DataFrame.from_dict(json_dict, orient="index")


def pandas_to_json_dict(df: pd.DataFrame) -> dict:
    json_str = df.to_json(orient='index')
    return json.loads(json_str)


def pandas_to_elasticsearch(df: pd.DataFrame, es: Elasticsearch, es_index: str,
                            es_type: str="doc", chunk_size=1000, verbose=True):

    def chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def send_chunk(json_string):
        r = es.bulk(json_string)
        return not(r.get("errors") is True)

    df_as_json = df.to_json(orient='records', lines=True)
    j_lines = df_as_json.split('\n')
    chunks_list = chunks(j_lines, chunk_size)
    chunks_len = math.ceil(len(j_lines) / chunk_size)
    if verbose:
        bar = Bar("Uploading", max=chunks_len, suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
    else:
        bar = None

    id_acum = 0
    for chunk in chunks_list:

        chunk_str = ""
        for json_document in chunk:
            jdict = json.loads(json_document)
            metadata = json.dumps({'index': {'_id': id_acum, '_index': es_index, '_type': es_type}})
            chunk_str += metadata + '\n' + json.dumps(jdict) + '\n'
            id_acum = id_acum + 1

        if chunk_str != "":
            if send_chunk(chunk_str):
                if verbose:
                    bar.next()
            else:
                raise Exception

    if verbose:
        print()


def elasticsearch_to_pandas(es: Elasticsearch, es_index: str, chunk_size=1000):

    parse_results = lambda data: {hit["_id"]: hit["_source"] for hit in data["hits"]["hits"]}
    get_chunk = lambda start: es.search(es_index, size=chunk_size, from_=start)

    hits = {}
    new_results = True
    i = 0
    while new_results:
        results = get_chunk(i)
        results = parse_results(results)

        i = i + chunk_size
        new_results = len(results) > 0
        hits.update(results)

    return json_dict_to_pandas(hits)
