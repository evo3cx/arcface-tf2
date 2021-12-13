import numpy as np
import tensorflow as tf
import os 
import cv2


from modules.evaluations import get_val_data, perform_val
from modules.models import ArcFaceModel
from modules.utils import set_memory_growth, load_yaml, l2_norm
from PIL import Image


def verify(compare, expect):
    """
    Function that verifies if the person on the "image_path" image is "identity".
    
    Arguments:
        image_path -- path to an image
        identity -- string, name of the person you'd like to verify the identity. Has to be an employee who works in the office.
        database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
        model -- your Inception model instance in Keras
    
    Returns:
        dist -- distance between the image_path and the image of "identity" in the database.
        door_open -- True, if the door should open. False otherwise.
    """
    ### START CODE HERE
    # Step 2: Compute distance with identity's image (≈ 1 line)
    dist = np.linalg.norm(compare-expect)
    print("dist", dist)
    # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
    if dist < 0.7:
        door_open = True
    else:
        door_open = False
    ### END CODE HERE        
    return dist, door_open

ckpt_path = tf.train.latest_checkpoint('./checkpoints/arc_mbv2_ccrop')
input_size = 112

model = ArcFaceModel(size=112,
                         backbone_type='MobileNetV2',
                         training=False)
model.load_weights(ckpt_path)
print(model.summary())

img_path = 'BruceLee.jpg'


# print(pred, "pred")
def img_to_encoding(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(112, 112))
    img = np.around(np.array(img) / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0)
    embedding = model.predict_on_batch(x_train)
    return embedding / np.linalg.norm(embedding, ord=2)

expect = img_to_encoding('BruceLee.jpg', model)
actual = img_to_encoding('./actual.jpeg', model)
min_dist = verify(expect, actual)
print(min_dist)

print('----------------------------------------')
reza1 = img_to_encoding('reza1.png', model)
reza2 = img_to_encoding('compare_1.png', model)
reza3 = img_to_encoding('reza3.png', model)
min_dist = verify(reza1, reza2)
print("1v2", min_dist)

min_dist = verify(reza1, reza3)
print("1v3", min_dist)