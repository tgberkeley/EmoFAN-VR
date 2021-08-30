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
import json


#### to run some tests on the landmarks
face_detector_kwargs = {
    # increase_min_score_thresh to minimise the chances of picking up 2 faces
    "min_score_thresh" : 0.80,
    "min_suppression_threshold" : 0.3
}

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu',
                                  flip_input=False, face_detector='blazeface',
                                  face_detector_kwargs=face_detector_kwargs)









class AffectNet(Dataset):

    def __init__(self, root_path, subset='train',
                 transform_image_shape=None, transform_image=None,
                 n_expression=5, verbose=1, cleaned_set=True):
        self.root_path = Path(root_path).expanduser()
        self.image_path = Path(root_path).expanduser()
        self.transform_image_shape = transform_image_shape
        self.transform_image = transform_image
        self.verbose = verbose
           
      ''' If do not have access to the json files below but do have access to the AffectNet 
          ".npy" files then uncomment code below and ignore opening json file code'''
        # all_data = {}
        #
        # all_files = os.listdir(self.root_path.joinpath('images/'))
        #
        #
        # if ".DS_Store" in all_files:
        #     all_files.remove(".DS_Store")
        #
        #
        # for file in all_files:
        #     image_name = os.path.splitext(file)[0]
        #
        #     file_path = str(self.root_path) + "/annotations/" + image_name
        #
        #     arousal = np.load(file_path + '_aro.npy')
        #     valence = np.load(file_path + '_val.npy')
        #     expression = np.load(file_path + '_exp.npy')
        #
        #     landmarks = np.load(file_path + '_lnd.npy')
        #     landmarks = np.reshape(landmarks, (-1, 2))
        #
        #     image_file = str(self.root_path) + "/images/" + file
        #
        #     image = io.imread(image_file)
        #     image = np.ascontiguousarray(image)
        #     predicted_landmarks = fa.get_landmarks(image)
        #     predicted_landmarks = np.array(predicted_landmarks).squeeze()
        #
        #     all_data[image_name] = {'valence': valence.item(),
        #                             'arousal': arousal.item(),
        #                             'expression': expression.item(),
        #                             'gt_landmarks': landmarks,
        #                             'my_landmarks': predicted_landmarks}
        #
        #

        
        # json file contain landmarks detected in preprocessing
        if subset == 'train':
            file = '/vol/bitbucket/tg220/data/training_data_affectnet.json'

        if subset == 'test':
            file = '/vol/bitbucket/tg220/data/affectnet_val_data.json'
            

        with open(file) as read_file:
            all_data = json.load(read_file)

            
        print(len(all_data.keys()))
        self.all_data = all_data
        self.image_keys = list(self.all_data.keys())

       
    def __len__(self):
        return len(self.image_keys)

    def __getitem__(self, index):
        
        # set to True if no face is detected
        ignore_bounding_box = False
        bounding_box = None
        key = self.image_keys[index]
        sample_image = self.all_data[key]


        key = 'images/' + key
        image_file = self.image_path.joinpath(key).as_posix()
        image_file = image_file + '.jpg'
        

        valence = torch.tensor([float(sample_image['valence'])], dtype=torch.float32)
        arousal = torch.tensor([float(sample_image['arousal'])], dtype=torch.float32)
        expression = torch.tensor([int(sample_image['expression'])], dtype=torch.long)
        
        ''' in Affectnet if -2 given as label it means label unknown therefore we give
            this a value of 0'''
        if valence == -2:
            valence = 0
        if arousal == -2:
            arousal = 0

        # pre-loaded landmarks from json file where we did landmark detection in preprocessing
        predicted_landmarks = sample_image['my_landmarks']
        predicted_landmarks = np.array(predicted_landmarks)

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
                bounding_box = [predicted_landmarks.min(axis=0)[0], predicted_landmarks.min(axis=0)[1],
                               predicted_landmarks.max(axis=0)[0], predicted_landmarks.max(axis=0)[1]]
                image, landmarks = self.transform_image_shape(image, bb= bounding_box)
                
            else:
                image, landmarks = self.transform_image_shape(image, bb=bounding_box)
            # Fix for PyTorch currently not supporting negative stric
            image = np.ascontiguousarray(image)
               
           '''uncomment this code to see the output of the above operations'''
            #img = Image.fromarray(image, 'RGB')
            #img.show()
            #sys.exit()
               
        
        if self.transform_image is not None:
            image = self.transform_image(image)


        return dict(valence=valence, arousal=arousal, expression=expression, image=image, au=[])












