import os
import sys
from pathlib import Path
import pickle
import numpy as np
import torch
import math
from torch.utils.data import Dataset
from skimage import io
from PIL import Image
import sys
import json
import cv2
import imutils
import face_alignment
from skimage import io
from Face_Occlusion import VR_patch


#### to run some tests on the landmarks
face_detector_kwargs = {
    # increase_min_score_thresh to minimise the chances of picking up 2 faces
    "min_score_thresh" : 0.85,
    "min_suppression_threshold" : 0.3
}
# video: 50 frame 4-10  38 frame 15-20 31-35  37 frame 0-10  16 frame 0 - 4+ more
# issue is we can get out ie two sets of landmarks for two different faces and no way to decipher
# which is the right one + eg AFEW video 16 ground truths look at i believe the wrong face
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu',
                                  flip_input=False, face_detector='blazeface',
                                  face_detector_kwargs=face_detector_kwargs)









class AffectNet(Dataset):

    def __init__(self, root_path, subset='test',
                 transform_image_shape=None, transform_image=None,
                 n_expression=5, verbose=1, cleaned_set=True):
        self.root_path = Path(root_path).expanduser()
        self.image_path = Path(root_path).expanduser()
        self.transform_image_shape = transform_image_shape
        self.transform_image = transform_image
        self.verbose = verbose

        all_data = {}

        all_files = os.listdir(self.root_path.joinpath('images/'))


        if ".DS_Store" in all_files:
            all_files.remove(".DS_Store")


        for file in all_files:
            image_name = os.path.splitext(file)[0]

            file_path = str(self.root_path) + "/annotations/" + image_name

            arousal = np.load(file_path + '_aro.npy')
            valence = np.load(file_path + '_val.npy')
            expression = np.load(file_path + '_exp.npy')

            landmarks = np.load(file_path + '_lnd.npy')
            landmarks = np.reshape(landmarks, (-1, 2))

            image_file = str(self.root_path) + "/images/" + file

            image = io.imread(image_file)
            image = np.ascontiguousarray(image)
            predicted_landmarks = fa.get_landmarks(image)
            predicted_landmarks = np.array(predicted_landmarks).squeeze()

            all_data[image_name] = {'valence': valence.item(),
                                    'arousal': arousal.item(),
                                    'expression': expression.item(),
                                    'gt_landmarks': landmarks,
                                    'my_landmarks': predicted_landmarks}

        print(len(all_data.keys()))

        self.all_data = all_data

        self.image_keys = list(self.all_data.keys())

        self.frame_keys_subset = []


       # self.frame_keys_subset = self.image_keys[130:210]
       # print(len(self.frame_keys_subset))

    def __len__(self):
        return len(self.image_keys)
        # return len(self.frame_keys)

    def __getitem__(self, index):
        # here index refers to the image code eg: '338/f60cd0dabca4c9cfcf2649cc99934d2570bddfd491d4420dac98bf49.jpg'
        ignore_bounding_box = False
        bounding_box = None
        key = self.image_keys[index]
        print(key)

        sample_image = self.all_data[key]


        key = 'images/' + key
        image_file = self.image_path.joinpath(key).as_posix()
        image_file = image_file + '.jpg'
        #print(image_file)

        valence = torch.tensor([float(sample_image['valence'])], dtype=torch.float32)
        arousal = torch.tensor([float(sample_image['arousal'])], dtype=torch.float32)
        expression = torch.tensor([float(sample_image['expression'])], dtype=torch.float32)

        if valence == -2:
            valence = 0
        if arousal == -2:
            arousal = 0

        gt_landmarks = sample_image['gt_landmarks']
        predicted_landmarks = sample_image['my_landmarks']


        if len(predicted_landmarks.shape) > 2:
            predicted_landmarks = predicted_landmarks[1,:,:]

        if predicted_landmarks.shape == ():
            ignore_bounding_box = True


        image = io.imread(image_file)
        image = np.ascontiguousarray(image)

        if ignore_bounding_box == False:
            VR_dimension = [20, 10]
            occluded_image = VR_patch(image, predicted_landmarks, VR_dim=VR_dimension)



        if self.transform_image_shape is not None:

            # uses predicted landmarks
            if ignore_bounding_box == False:
            #    bounding_box = [predicted_landmarks.min(axis=0)[0], predicted_landmarks.min(axis=0)[1],
            #                    predicted_landmarks.max(axis=0)[0], predicted_landmarks.max(axis=0)[1]]

                image, landmarks = self.transform_image_shape(occluded_image, bb= bounding_box)
            else:
                image, landmarks = self.transform_image_shape(image, bb=bounding_box)
            # Fix for PyTorch currently not supporting negative stric
            image = np.ascontiguousarray(image)

            #img = Image.fromarray(image, 'RGB')
            #img.show()
            #sys.exit()
        ########################


        if self.transform_image is not None:
            image = self.transform_image(image)


        return dict(valence=valence, arousal=arousal, expression=1, image=image, au=[])











