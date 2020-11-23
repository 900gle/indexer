import tensorflow as tf
import tensorflow_hub as hub

# outputs = module(dict(images=daangn_profile_image), signature="image_feature_vector", as_dict=True)
# target_image = outputs['default']
img = "./images/origin.png"
#
# module = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2", input_shape=(None,224,224,3), dtype=tf.float32, name='inputs')
module = hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/1")


features =  module(dict(images=img), signature="image_feature_vector", signature_outputs_as_dict=True)

# target_image = features['default']
# print(target_image)
#
#
print(features.shape)


print(('Your TensorFlow version: {0}').format(tf.__version__))


# m = tf.keras.Sequential([
#     hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/4",
#                    trainable=False),  # Can be True, see below.
#     tf.keras.layers.Dense(num_classes, activation='softmax')
# ])
# m.build([None, 96, 96, 3])  # Batch input shape.


# outputs = module(dict(images=daangn_profile_image), signature="image_feature_vector", as_dict=True)
# target_image = outputs['default']