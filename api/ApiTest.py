import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json

from flask import Flask
from flask_restful import reqparse, abort, Api, Resource

app = Flask(__name__)
api = Api(app)

filePath = "/Users/doo/project/opencv/web/src/main/resources/static/images/"
tempPath = "/Users/doo/project/opencv/common/temp/"


def load_img(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize_with_pad(img, 224, 224)
    img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    return img


parser = reqparse.RequestParser()


def getImageFeatureVectors(path, fileName):
    module_handle = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4"
    module = hub.load(module_handle)
    img = load_img(path + fileName)
    features = module(img)
    feature_set = np.squeeze(features)
    return feature_set


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Api(Resource):
    def get(self, fileName):
        return {'vectors': json.dumps(getImageFeatureVectors(tempPath, fileName), cls=NumpyEncoder)}

    def post(self, fileName):
        args = parser.parse_args()
        return {'vectors': json.dumps(getImageFeatureVectors(tempPath, fileName), cls=NumpyEncoder)}


class Vector(Resource):
    def get(self, fileName):
        return {'vectors': json.dumps(getImageFeatureVectors(filePath, fileName), cls=NumpyEncoder)}

    def post(self, fileName):
        args = parser.parse_args()
        return {'vectors': json.dumps(getImageFeatureVectors(filePath, fileName), cls=NumpyEncoder)}


class TestApp(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument('value', type=str, required=True, help="This field cannot be blank.")

    def get(self, fileName):
        return {'tests': fileName}

    def post(self, fileName):
        data = TestApp.parser.parse_args()
        return {'tests': data['value']}


api.add_resource(TestApp, '/test/<string:fileName>')

api.add_resource(Vector, '/vectors/<string:fileName>')
api.add_resource(Api, '/api/<string:fileName>')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8088)
