import numpy as np
import cv2
#import face_utils
import imutils
from imutils import face_utils




def scale_VR(dim, shape_):
    '''
    Function to scale VR headset dimension based on the distance between 2 temporal bones
      Parameters:
        dim:list [w, h], width and height of the VR headset
        shape: ndarray with shape (68, 2)
      return:
        w: integer, scaled width of the VR headset
        h: integer, scaled height of the VR headset
    '''
    ratio = dim[1]/dim[0]
    right_temple = shape_[0]
    left_temple = shape_[16]
    dY = right_temple[1] - left_temple[1]
    dX = right_temple[0] - left_temple[0]
    dist = np.sqrt(dY ** 2 + dX ** 2)
    w = int(dist)
    h = int(ratio * w)
    return w, h

def find_EyeCentre(shape):
    '''
    Function to determine centre of each detected eyes
      Parameters:
        shape: ndarray with shape (68, 2)
      return:
        rcent: ndarray with shape (2,), right eye centre x and y coordinates
        lcent: ndarray with shape (2,), left eye centre x and y coordinates
    '''
    # extract the left and right eye x and y coordinates
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye']
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye']
    rightEyePts = shape[rStart:rEnd]
    leftEyePts = shape[lStart:lEnd]

    # compute the center of each eye
    rcent = rightEyePts.mean(axis=0).astype('int')
    lcent = leftEyePts.mean(axis=0).astype('int')

    return rcent, lcent

def get_VR_points(m, d):
    '''
    Function to compute 4 corner points of VR headset based on eye position
      Parameters:
        m: tuple(x, y), coordinate of centre between two eyes
        d: list [w, h], width and height of VR headset
      return:
        ur_vr: tuple(x, y), coordinate of upper-right point in VR headset
        br_vr: tuple(x, y), coordinate of bottom-right point in VR headset
        bl_vr: tuple(x, y), coordinate of bottom-left point in VR headset
        ul_vr: tuple(x, y), coordinate of upper-left point in VR headset
    '''
    VR_w = d[0] // 2
    VR_h = d[1] // 2
    bl_vr = m[0] + VR_w, m[1] + VR_h
    ul_vr = m[0] + VR_w, m[1] - VR_h
    ur_vr = m[0] - VR_w, m[1] - VR_h
    br_vr = m[0] - VR_w, m[1] + VR_h

    return ur_vr, br_vr, bl_vr, ul_vr


def rotate_pts(rcent, lcent, pts, m):
    '''
    Function to rotate points around a midpoint
      Parameters:
        rcent: ndarray with shape (2,), right eye centre x and y coordinates
        lcent: ndarray with shape (2,), left eye centre x and y coordinates
        pts: tuple of tuples (4,2), position of VR headset corner points
        m: tuple(x, y), coordinate of centre between two eyes
      return:
        rotated: list of rotated points around a midpoint
    '''
    dY = rcent[1] - lcent[1]
    dX = rcent[0] - lcent[0]
    theta = np.arctan(dY /dX)

    rotation_matrix = np.array(( (np.cos(theta), -np.sin(theta)),
                                 (np.sin(theta), np.cos(theta)) ))

    rotated =[]
    for i in range(len(pts)):
        v = np.array(pts[i]) - np.array(m)
        rotated.append(tuple(rotation_matrix.dot(v).astype(int) + m))

    return rotated


def VR_patch(x, shapes, VR_dim=[20, 10]):
    '''
          Function to apply VR patch  on image
            Parameters:
              x: ndarray with shape (number of images, image width in pixel, image height in pixel), grayscale
              shapes: ndarray with shape (number of images, 68, 2), landmark coordinates
              VR_dim: VR headset dimension [w, h]
    '''
    image = x
    
    # compute VR headset position on the face
    VR_dim_scaled = scale_VR(VR_dim, shapes)
    # determine each eye's centre
    rightEyeCenter, leftEyeCenter = find_EyeCentre(shapes)
    # find the centre between two eyes
    midpoint = (rightEyeCenter[0] + leftEyeCenter[0]) // 2, (rightEyeCenter[1] + leftEyeCenter[1]) // 2
    # compute VR points
    VR_pts = get_VR_points(midpoint, VR_dim_scaled)
    # determine VR headset aligned with eyes' position
    VR_pts_rotated = rotate_pts(rightEyeCenter, leftEyeCenter, VR_pts, midpoint)
    # overlay VR headset patch on the image
    cv2.fillPoly(image, [np.int32(tuple(VR_pts_rotated))], 1, 255)
    
    return image
