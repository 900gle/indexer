import json
import time

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

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

    # origin_image = "./images/origin.png"
    origin_image = "./images/target3.png"
    # origin_image = "./images/image_test1.png"
    img = load_img(origin_image)
    features = module(img)
    feature_set = np.squeeze(features)
    print(feature_set)
    print("vector dmis : " , len(feature_set))
    return feature_set


# get_image_feature_vectors()


docs = []

es = Elasticsearch(
    ['localhost'],
    http_auth=('elastic', 'dlengus'),
    scheme="http",
    port=9200,
)

feature = get_image_feature_vectors()
doc = {
    "feature": feature,
    "image_name": "target3.png",
    "image_id": "3",
}

print(doc)
res = es.index(index="tensor_images_test", id=3, body=doc)
print(res['result'])

es.indices.refresh(index="tensor_images_test")

# res = es.get(index="tensor_images", id='hTex1HUBmn0MvYB6KMOT')
# print(res['_source'])