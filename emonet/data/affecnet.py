from pathlib import Path 
import pickle
import numpy as np 
import torch
import math
from torch.utils.data import Dataset
from skimage import io
from PIL import Image
import sys

class AffectNet(Dataset):
    _expressions = {0: 'neutral', 1:'happy', 2:'sad', 3:'surprise', 4:'fear', 5:'disgust', 6:'anger', 7:'contempt', 8:'none'}
    _expressions_indices = {8: [0, 1, 2, 3, 4, 5, 6, 7], 
                            5: [0, 1, 2, 3, 6]}
    
    def __init__(self, root_path, subset='test',
                 transform_image_shape=None, transform_image=None,
                 n_expression=5, verbose=1, cleaned_set=True):
        self.root_path = Path(root_path).expanduser()
        self.subset = subset
        self.image_path = self.root_path.joinpath(subset)
        self.transform_image_shape = transform_image_shape
        self.transform_image = transform_image
        self.verbose = verbose

        #if cleaned_set and (subset not in ['test', 'val']):
        #    raise ValueError('cleaned_set can only be set to True for the val or test set, train has not been cleaned')
        self.cleaned_set = cleaned_set

        if n_expression not in [5, 8]:
            raise ValueError(f'n_expression should be either 5 or 8, but got n_expression={n_expression}')
        self.n_expression = n_expression

        #self.pickle_path = self.root_path.joinpath('pickles/',f'{subset}_fullpath.pkl')
        #with open(self.pickle_path, 'br') as f:
            #data = pickle.load(f)
        #self.data = data
        #print(self.data['421/2864f1dd2361dee9c627e42ec71d484c4425625e0045d2176245adad.jpg'])
        #print(self.data.keys())
        #print(self.data['338/f60cd0dabca4c9cfcf2649cc99934d2570bddfd491d4420dac98bf49.jpg'])
        #print(self.data)
        a = np.load('AffectNet_val_set/annotations/421_lnd.npy')
        print(a)
#'expression': '8', 'valence': '0.120983', 'arousal': '-0.464576',
        # the keys are the image names (name.ext)
        self.keys = []
        self.skipped = {'other':[], 'pt_pt_error':[], 'expression':[], 'cleaned':[]}

        # List of each expression to generate weights
        expressions = []
        for key, value in data.items():
            if key == 'folder':
                continue
            if (int(value['expression']) not in self._expressions_indices[self.n_expression]):
                self.skipped['expression'].append(key)
                continue
            if self.cleaned_set and (not value['expression_correct']):
                self.skipped['cleaned'].append(key)
                continue

            expression = int(value['expression'])
            if self.cleaned_set:
                #Automatic cleaning : expression has to match the valence and arousal values
                valence = float(value['valence'])
                arousal = float(value['arousal'])
                intensity = math.sqrt(valence**2+arousal**2)

                if expression == 0 and intensity>=0.2:
                    self.skipped['other'].append(key)
                    continue
                elif expression == 1  and (valence<=0 or intensity<=0.2):
                    self.skipped['other'].append(key)
                    continue           
                elif expression == 2  and (valence>=0 or intensity<=0.2):
                    self.skipped['other'].append(key)
                    continue
                elif expression == 3  and (arousal<=0 or intensity<=0.2):
                    self.skipped['other'].append(key)
                    continue
                elif expression == 4  and (not(arousal>=0 and valence<=0) or intensity<=0.2):
                    self.skipped['other'].append(key)
                    continue
                elif expression == 5  and (valence>=0 or intensity<=0.3):
                    self.skipped['other'].append(key)
                    continue
                elif expression == 6  and (arousal<=0 or intensity<=0.2):
                    self.skipped['other'].append(key)
                    continue
                elif expression == 7  and (valence>=0 or intensity<=0.2):
                    self.skipped['other'].append(key)
                    continue
 
                if self.n_expression == 5 and expression == 6:
                    expression = 4
            expressions.append(expression)
            self.keys.append(key)

        expressions = np.array(expressions)
        self.sample_per_class = {label:np.sum(expressions == label) for label in np.unique(expressions)}
        self.expression_weights = np.array([1./self.sample_per_class[e] for e in expressions])
        self.average_per_class = int(np.mean(list(self.sample_per_class.values())))

        if self.verbose:
            skipped = sum([len(self.skipped[key]) for key in self.skipped])
            msg = f' --  {len(self.keys)} images, skipped {len(self.skipped)} images ({len(self.skipped["pt_pt_error"])} with large errors).'
            print(msg)
            print(f'Samples per class : {self.sample_per_class}')

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        # here index refers to the image code eg: '338/f60cd0dabca4c9cfcf2649cc99934d2570bddfd491d4420dac98bf49.jpg'
        key = self.keys[index]
        sample_data = self.data[key]

        image_file = self.image_path.joinpath(key).as_posix()

        valence = torch.tensor([float(sample_data['valence'])], dtype=torch.float32)
        arousal = torch.tensor([float(sample_data['arousal'])], dtype=torch.float32)
        expression = int(sample_data['expression'])
    
        if self.n_expression == 5 and expression == 6:
            expression = 4

        # will need to change this to sample_data['landmarks'] to use our own landmarks instead of emofans
        landmarks = sample_data['landmarks_fan']
        
        if isinstance(landmarks, list):
            landmarks = np.array(landmarks)
        image = io.imread(image_file)


        if self.transform_image_shape is not None:
            bounding_box = [landmarks.min(axis=0)[0], landmarks.min(axis=0)[1],
                            landmarks.max(axis=0)[0], landmarks.max(axis=0)[1]]
            #image, landmarks = self.transform_image_shape(image, shape=landmarks)
            image, landmarks = self.transform_image_shape(image, bb=bounding_box)
            # Fix for PyTorch currently not supporting negative stric
            image = np.ascontiguousarray(image)

            img = Image.fromarray(image, 'RGB')
            img.show()
            #
            sys.exit()

        if self.transform_image is not None:
            image = self.transform_image(image)

        return dict(valence=valence, arousal=arousal, expression=expression, image=image, au=[])


#Each key and value pair in the dictionary:

#  '957/66f024fee5e8dd72e9807787166ea5923d99ee5b03db1e1ac5bf305c.jpg':
#  {'faceX': '21', 'faceY': '21', 'face_width': '145', 'face_height': '145', 'landmarks': array([[ 40.35,  85.68],...
#  [ 93.11, 145.83]], dtype=float32), 'expression': '3', 'valence': '-0.275842', 'arousal': '0.590399',
#  'landmarks_fan': array([[ 41.63452515,  90.50995066], ..... [ 93.18876611, 146.1885309 ]]),
#  'bbox_fan': array([ 36.45433  ,  10.316315 , 156.10971  , 181.01443  ,   0.9999987],
#       dtype=float32), 'landmarks_fan_avg_score': 0.8051564,
#       'augmented_version': ['66f024fee5e8dd72e9807787166ea5923d99ee5b03db1e1ac5bf305c_p29_y38_r0_augmented.jpg'],
#       'yaw': -0.24767218198033186, 'pitch': 0.15320588468903834,
#       'roll': 0.14587145668361087, 'expression_correct': True, 'augmented': False},