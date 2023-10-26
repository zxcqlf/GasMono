# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from .mono_dataset import MonoDataset


class NYUDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(NYUDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        self.K = np.array([[0.81, 0, 0.51, 0],
                           [0, 1.08, 0.53, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (640, 480)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):

        return False

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class NYURAWDataset(NYUDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(NYURAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, '.jpg')
        image_path = os.path.join(
            self.data_path, folder, f_str)
        return image_path
        
    def get_pose(self, folder, frame_index, side, do_flip):
        f_str = "{:04d}{}".format(frame_index, '.npy')
        pose_path = os.path.join(
            './splits/nyu/cpose/', folder, f_str)
        pose = np.load(pose_path)
        return pose

class NYUTestDataset(NYUDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(NYUTestDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:04d}{}".format(frame_index, '.png')
        image_path = os.path.join(
            self.data_path, folder, f_str)
        return image_path

