# need to turn each video into a series of images first and store them in a large folder with labels

# then need a large folder with all the annotations for each frame so that they can be matched up
# ideally a dictionary with key for the video then that value has keys for each frame
# with the values for that being the link to the image file, the valence and the arousal

# could create a json file with this dictionary exactly as I need it (might need two, one with below and
# the other with a list of all the video names to use as keys)

# something like self.all_data contains the video number as original key and within this the values
# are all the frames. {'001': {'00000': {'arousal': 5.0, 'valence': 0.0, 'landmarks': [[181.4181259729673, 170.1050112037773],
# [181.8897679358958, 186.95041563929297] .....

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

# 76-30-640x280_1580.jpg question the last frames annotations for this one (is it missing a -ve sign?????)

# look at end opf video 81 last frame 8440 i beleive

# 112-30-640x360_5400 has 5 faces detected 50x32 from 6000 frames



# will be bugs for eg video 55 and 74 and has left and right annotations which will lead to problems
# remove these two

all_data = {}


all_files = os.listdir("AffWild2_VA_annotations/Validation_Set/")
all_files.remove("video55_left.txt")
all_files.remove("video55_right.txt")
all_files.remove("video74_left.txt")
all_files.remove("video74_right.txt")
print(len(all_files))

for file in all_files:

    video_name = os.path.splitext(file)[0]
    all_data[video_name] = {}

    file_path = "AffWild2_VA_annotations/Validation_Set/" + file
    with open(file_path) as f:
        lines = f.readlines()[1:]
        frame_number = 0
        for line in lines:

            line = line.rstrip("\n")

            VA = line.split(",")
            all_data[video_name][frame_number] = {'valence' : VA[0],
                                                  'arousal' : VA[1]}
            frame_number +=1


print(len(all_data.keys()))



# this face detection is from https://github.com/1adrianb/face-alignment/blob/master/README.md

face_detector_kwargs = {
    # increase_min_score_thresh to minimise the chances of picking up 2 faces
    "min_score_thresh" : 0.85,
    "min_suppression_threshold" : 0.3
}
'''video: 50 frame 4-10  38 frame 15-20 31-35  37 frame 0-10  16 frame 0 - 4+ more
   issue is we can get out ie two sets of landmarks for two different faces and no way to decipher
   which is the right one'''
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu',
                                  flip_input=False, face_detector='blazeface',
                                  face_detector_kwargs=face_detector_kwargs)




class AffWild_2(Dataset):

    def __init__(self, root_path, subset='test',
                 transform_image_shape=None, transform_image=None,
                 n_expression=5, verbose=1, cleaned_set=True):
        self.root_path = Path(root_path).expanduser()
        self.image_path = Path(root_path).expanduser()
        self.transform_image_shape = transform_image_shape
        self.transform_image = transform_image
        self.verbose = verbose



        # self.all_data contains the video number as original key and within this the values
        # are all the frames. {'001': {'00000': {'arousal': 5.0, 'valence' : 1.0,}, '00001': ....
        self.all_data = all_data

        self.frame_keys = []

        self.frame_keys_subset = []

        # all the videos we are going to run through
        for video in self.all_data.keys():

            frames = self.all_data[video].keys()

            for frame in frames:
                if int(frame) % 20 == 0:
                    image_root = video + '_' + str(frame)
                    self.frame_keys.append(image_root)



        print(len(self.frame_keys))
        self.frame_keys_subset = self.frame_keys[6000:12000]
        print(len(self.frame_keys_subset))

    def __len__(self):
        return len(self.frame_keys_subset)
        #return len(self.frame_keys)

    def __getitem__(self, index):
        # here index refers to the image code eg: '338/f60cd0dabca4c9cfcf2649cc99934d2570bddfd491d4420dac98bf49.jpg'
        ignore_bounding_box = False
        bounding_box = None

        key = self.frame_keys_subset[index]
        #key = self.frame_keys[index]
        #print(self.keys)
        #print(self.frame_keys)

        print(key)


        x = key.split("_")
        # as eg video 86 has 3 parts
        if len(x) == 3:
            y = x[0] + "_" + x[1]
            z = x[2]

        else:
            y = x[0]
            z = x[1]

        sample_video = self.all_data[y]
        #print(sample_video.keys())

        #frame_number = f"{int(x[1]):05}"
        frame_number = int(z)

        sample_data = sample_video[frame_number]
        #print(sample_data.keys())
        #print(sample_data)


        image_file = self.image_path.joinpath(key).as_posix()
        image_file = image_file + '.jpg'



        valence = torch.tensor([float(sample_data['valence'])], dtype=torch.float32)
        arousal = torch.tensor([float(sample_data['arousal'])], dtype=torch.float32)

        #landmarks = sample_data['landmarks']

        # will need to change this to sample_data['landmarks'] to use our own landmarks instead of emofans
        #landmarks = sample_data['landmarks_fan']

        #if isinstance(landmarks, list):
        #landmarks = np.array(landmarks)


        image = io.imread(image_file)
        image = np.ascontiguousarray(image)


        predicted_landmarks = fa.get_landmarks(image)

        # still need to fix if finds 2 faces
        predicted_landmarks = np.array(predicted_landmarks).squeeze()
        print(predicted_landmarks.shape)
        # if it predicts 2 bounding boxes as it detects 2 faces

        if predicted_landmarks.shape == (2,68,2):
            predicted_landmarks = predicted_landmarks[1,:,:]
        if predicted_landmarks.shape == (3,68,2):
            predicted_landmarks = predicted_landmarks[1,:,:]
        #print(predicted_landmarks)
        # this is if no face is detected, could look to predict 0, 0
        if predicted_landmarks.shape == ():
            ignore_bounding_box = True




        if self.transform_image_shape is not None:
            
            # uses predicted landmarks
            if ignore_bounding_box == False:
                bounding_box = [predicted_landmarks.min(axis=0)[0], predicted_landmarks.min(axis=0)[1],
                                predicted_landmarks.max(axis=0)[0], predicted_landmarks.max(axis=0)[1]]



            image, landmarks = self.transform_image_shape(image, bb= bounding_box)
            # Fix for PyTorch currently not supporting negative stric
            image = np.ascontiguousarray(image)
            
            '''uncomment this code to see the output of the above operations'''
            #img = Image.fromarray(image, 'RGB')
            #img.show()
            #sys.exit()
        


        if self.transform_image is not None:
            image = self.transform_image(image)


        return dict(valence=valence, arousal=arousal, expression=1, image=image, au=[])



