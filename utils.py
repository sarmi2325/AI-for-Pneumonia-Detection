import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)  # (224,224,3)
    img_array = np.expand_dims(img_array, axis=0)               # (1,224,224,3)
    img_array = preprocess_input(img_array)                      # preprocess
    img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)  # convert to tensor
    return img_array



def make_gradcam_heatmap(img_array, model, last_conv_layer_name='top_conv'):
    base_model = model.get_layer('efficientnetb0')
    last_conv_layer = base_model.get_layer(last_conv_layer_name)
    inputs = model.input

    last_conv_layer_output = last_conv_layer.output


    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[last_conv_layer.output, base_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, base_preds = grad_model(img_array, training=False)
        
        
        x = base_preds
        x = model.get_layer('global_average_pooling2d')(x)
        x = model.get_layer('dropout')(x)
        preds = model.get_layer('dense')(x)
        
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def superimpose_heatmap(heatmap, original_image, alpha=0.4):
    img = np.array(original_image.resize((224, 224)))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return superimposed
