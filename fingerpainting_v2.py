import time
import numpy as np
import os
import matplotlib.pyplot as plt
from mrcnn.config import Config
from mrcnn import model as modellib, utils
import cv2

specific_model = 'fingerpainting'
in_path = 'trainingdata/labeled_palm/batch1/'
model_weights = 'mask_rcnn_fingerpainting_0300.h5'

class FingerpaintingConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = specific_model
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1 # Batch size to train on
    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # Background + ROI's
    STEPS_PER_EPOCH = len(os.listdir(in_path))
    print(STEPS_PER_EPOCH)
    DETECTION_MIN_CONFIDENCE = 0.90
    # MAX_GT_INSTANCES = 1
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    # TRAIN_ROIS_PER_IMAGE = 32
    # BACKBONE = "resnet50"

print('Loading model...')
model = modellib.MaskRCNN(mode="inference", config=FingerpaintingConfig(),
                                 model_dir='')
model.load_weights(model_weights, by_name=True)
print('Model loaded...')


print('Getting mask...')
with open('smudgepts.txt','r') as inf:
    pts = eval(inf.read())
x = pts['x']
y = pts['y']
pts = list(zip(x,y))
pts = np.array(pts)
image = cv2.imread('starterimage.jpg')
mask = np.ones(image.shape[:2])
mask = cv2.fillPoly(mask, [pts], 0)
og_mask = np.copy(mask)




print('Warming up the model...')
hands = model.detect([image], verbose=False)
print('Warmed up')

# for f in os.listdir(in_path):
#     print(f)
#     s_time = time.time()
#
#     image = cv2.imread(in_path + f)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     hands = model.detect([image], verbose=False)
#
#     print(time.time() - s_time)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    _, frame = cap.read()
    cp = np.copy(frame)
    cp[np.where(mask == 0)] = (255,0,0)

    hands = model.detect([frame], verbose=False)
    r_hands = hands[0]
    hand_instance = r_hands['rois']
    print(len(hand_instance))

    if len(hand_instance) > 0:
        mask_dat = r_hands['masks']
        hand_mask = mask_dat[:,:,0]
        # print(hands_mask)
        mask[np.where(hand_mask == 1)] = 1

    cv2.imshow('', cp)
    key = cv2.waitKey(1)

    if key == 32:  # 32 is space key on my computer.
        mask = og_mask



# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     raise IOError("Cannot open webcam")
#
# pic_array = np.array([])
# n = 1
# while True:
#     _, frame = cap.read()
#
#     cv2.imshow('', frame)
#     key = cv2.waitKey(1)
#     cv2.imwrite('trainingdata/'+str(n)+'.jpg', frame)
#     n += 1
#     if key == 32:  # 32 is space key on my computer.
#         break
#
# cap.release()
# cv2.destroyAllWindows()

