import numpy as np
import cv2
import tensorflow as tf

def grad_cam(model, image_array, layer_name):
    """
    Compute Grad-CAM for the given model and image.
    (Simplified implementation.)
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        loss = predictions[:, tf.argmax(predictions[0])]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    cam = conv_outputs @ pooled_grads[..., tf.newaxis]
    cam = tf.squeeze(cam).numpy()
    cam = np.maximum(cam, 0)
    cam = cam / (np.max(cam) + 1e-8)
    return cam

def grad_cam_plus(model, image_array, layer_name):
    """
    Compute Grad-CAM++ for the given model and image.
    (Currently using the same implementation as Grad-CAM; extend as needed.)
    """
    return grad_cam(model, image_array, layer_name)

def overlay_heatmap(image, heatmap, alpha=0.4):
    """
    Overlay the heatmap onto the original image.
    """
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    overlayed = cv2.addWeighted(image_bgr, alpha, heatmap, 1 - alpha, 0)
    overlayed = cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB)
    return overlayed
