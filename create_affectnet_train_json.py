import os
import numpy as np
from skimage import io
import face_alignment
import json


#### to run some tests on the landmarks
face_detector_kwargs = {
    # increase_min_score_thresh to minimise the chances of picking up 2 faces
    "min_score_thresh" : 0.85,
    "min_suppression_threshold" : 0.3
}
# video: 50 frame 4-10  38 frame 15-20 31-35  37 frame 0-10  16 frame 0 - 4+ more
# issue is we can get out ie two sets of landmarks for two different faces and no way to decipher
# which is the right one + eg AFEW video 16 ground truths look at i believe the wrong face
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda',
                                  flip_input=False, face_detector='blazeface',
                                  face_detector_kwargs=face_detector_kwargs)

root_path = "/vol/bitbucket/tg220/data/train_set/"

all_data = {}

all_files = os.listdir(root_path + 'images/')
print(len(all_files))

if ".DS_Store" in all_files:
    all_files.remove(".DS_Store")

counter = 0
for file in all_files:
    image_name = os.path.splitext(file)[0]
    print(counter)
    counter += 1
    file_path = str(root_path) + "/annotations/" + image_name

    arousal = np.load(file_path + '_aro.npy')
    valence = np.load(file_path + '_val.npy')
    expression = np.load(file_path + '_exp.npy')

    landmarks = np.load(file_path + '_lnd.npy')
    landmarks = np.reshape(landmarks, (-1, 2))
    landmarks = landmarks.tolist()

    image_file = str(root_path) + "/images/" + file

    image = io.imread(image_file)
    image = np.ascontiguousarray(image)
    predicted_landmarks = fa.get_landmarks(image)
    predicted_landmarks = np.array(predicted_landmarks).squeeze()
    predicted_landmarks = predicted_landmarks.tolist()

    all_data[image_name] = {'valence': valence.item(),
                            'arousal': arousal.item(),
                            'expression': expression.item(),
                            'gt_landmarks': landmarks,
                            'my_landmarks': predicted_landmarks}


with open('/vol/bitbucket/tg220/data/training_data_affectnet2.json', 'w') as fp:
    json.dump(all_data, fp, sort_keys=True, indent=4)

