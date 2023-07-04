import cv2
import numpy as np
from keras.models import load_model


# Classify fresh/rotten
def print_fresh_rotten(res):
    threshold_fresh = 0.2                # set according to standards
    if res < threshold_fresh:            # implement two thresholds later, for med,rotten and fresh ident
        print("The item is FRESH!")         
    else:
        print("The item is ROTTEN")


def pre_proc_img(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    img = cv2.resize(img, (100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Preprocess the image
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def evaluate_rotten_vs_fresh(image_path):
    # Load the trained model
    model = load_model('trained-freshness-model.h5')

    # Read and process and predict
    prediction = model.predict(pre_proc_img(image_path))

    return prediction[0][0]


# Example usage:
img_path = 'image-to-eval.png'
is_rotten = evaluate_rotten_vs_fresh(img_path)
print(f'Prediction: {is_rotten}',print_fresh_rotten(is_rotten))
