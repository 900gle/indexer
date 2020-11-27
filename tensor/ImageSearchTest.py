import time

from elasticsearch import Elasticsearch

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

def load_img(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize_with_pad(img, 224, 224)
    img = tf.image.convert_image_dtype(img,tf.float32)[tf.newaxis, ...]
    return img

def get_image_feature_vectors() :

    module_handle = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4"
    module = hub.load(module_handle)

    origin_image = "./images/origin.png"
    img = load_img(origin_image)
    features = module(img)
    feature_set = np.squeeze(features)
    print(feature_set)
    print("vector dmis : " , len(feature_set))
    return feature_set

es = Elasticsearch(
    ['localhost'],
    http_auth=('elastic', 'dlengus'),
    scheme="http",
    port=9200,
)

script_query = {
    "script_score": {
        "query": {"match_all": {}},
        "script": {
            "source": "cosineSimilarity(params.query_vector, 'feature')",
            "params": {"query_vector": get_image_feature_vectors()}
        }
    }
}

search_start = time.time()
response = es.search(
    index="tensor_images_test",
    body={
        "query": script_query,
        "_source": {"includes": ["image_name", "image_id"]}
    }
)

for hit in response["hits"]["hits"]:
    print("id: {}, score: {}".format(hit["_id"], hit["_score"]))
    print(hit["_source"])
    print()

