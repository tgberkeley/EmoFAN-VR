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

# this face detection is from https://github.com/1adrianb/face-alignment/blob/master/README.md

face_detector_kwargs = {
    # increase_min_score_thresh to minimise the chances of picking up 2 faces
    "min_score_thresh" : 0.7,
    "min_suppression_threshold" : 0.3
}
# video: 50 frame 4-10  38 frame 15-20 31-35  37 frame 0-10  16 frame 0 - 4+ more
# issue is we can get out ie two sets of landmarks for two different faces and no way to decipher
# which is the right one + eg AFEW video 16 ground truths look at i believe the wrong face
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu',
                                  flip_input=False, face_detector='blazeface',
                                  face_detector_kwargs=face_detector_kwargs)



class AffectNet(Dataset):

    def __init__(self, root_path, subset=None,
                 transform_image_shape=None, transform_image=None,
                 n_expression=5, verbose=1, cleaned_set=True):
        self.root_path = Path(root_path).expanduser()
        self.image_path = Path(root_path).expanduser()
        self.transform_image_shape = transform_image_shape
        self.transform_image = transform_image
        self.verbose = verbose


        # need to get the image

        # works now to get from loads and loads of images, therefore we need to get it to
        # loop through all our video folder images and have as a long list of images


        # self.all_data contains the video number as original key and within this the values
        # are all the frames. {'001': {'00000': {'arousal': 5.0, 'landmarks': [[181.4181259729673, 170.1050112037773],
        # [181.8897679358958, 186.95041563929297] .....
        self.all_data = {}

        self.frame_keys = []
        self.keys = []
#         if subset == "train":
#             film_start = 0
#             film_end = 500

#         if subset == "test":
#             film_start = 500
#             film_end = 600
        film_start = 0
        film_end = 600
        for i in range(film_start, film_end):

            with open(root_path + f"{i+1:03d}" + "/" + f"{i+1:03d}" + ".json", "r") as read_file:
                data = json.load(read_file)
            self.data = data
            video_num = self.data['video_id']
            self.all_data[video_num] = {}


            #print(data.keys())
            # self.keys are all the 50 videos
            self.keys.append(data['video_id'])

            frames = self.data['frames']
            self.all_data[video_num] = frames


            self.frames = frames
            #print(frames.keys())
            for key, value in frames.items():
                if key !=None:
                    image_root = self.data['video_id'] + "/" + key
                    self.frame_keys.append(image_root)

        print(len(self.frame_keys))

    def __len__(self):
        return len(self.frame_keys)

    def __getitem__(self, index):
        # here index refers to the image code eg: '338/f60cd0dabca4c9cfcf2649cc99934d2570bddfd491d4420dac98bf49.jpg'
        ignore_bounding_box = False
        bounding_box = None

        key = self.frame_keys[index]
        #print(self.keys)
        #print(self.frame_keys)

        print(key)

        x = key.split("/")

        sample_video = self.all_data[x[0]]
        #print(sample_video.keys())

        sample_data = sample_video[x[1]]
        #print(sample_data.keys())

        image_file = self.image_path.joinpath(key).as_posix()
        image_file = image_file + '.png'

        valence = torch.tensor([float(sample_data['valence'])], dtype=torch.float32)
        # so that we get valence and arousal between -1 and 1
        valence = valence / 10
        arousal = torch.tensor([float(sample_data['arousal'])], dtype=torch.float32)
        arousal = arousal / 10

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
        #print(predicted_landmarks.shape)
        # if it predicts 2 or more bounding boxes as it detects 2 or more faces

        if len(predicted_landmarks.shape) > 2:
            predicted_landmarks = predicted_landmarks[1,:,:]

        if predicted_landmarks.shape == ():
            ignore_bounding_box = True

        if ignore_bounding_box == False:
            VR_dimension = [20, 10]
            occluded_image = VR_patch(image, predicted_landmarks, VR_dim=VR_dimension)



        if self.transform_image_shape is not None:
            # uses ground truth landmarks
            #bounding_box = [landmarks.min(axis=0)[0], landmarks.min(axis=0)[1],
                            #landmarks.max(axis=0)[0], landmarks.max(axis=0)[1]]
            # uses predicted landmarks
            if ignore_bounding_box == False:
                bounding_box = [predicted_landmarks.min(axis=0)[0], predicted_landmarks.min(axis=0)[1],
                                predicted_landmarks.max(axis=0)[0], predicted_landmarks.max(axis=0)[1]]

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

        # img = image.mul(255).byte()
        # img = img.cpu().numpy().transpose((1, 2, 0))
        # img = Image.fromarray(img, 'RGB')
        # img.show()
        #


        return dict(valence=valence, arousal=arousal, expression=1, image=image, au=[])


