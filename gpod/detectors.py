# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import os
import warnings

import numpy
import numpy.random

import scipy.ndimage

from .markobjects import Mark2D


class _Detector(object):
    """
    Base detector object.
    """

    def __init__(self, loader, descriptor, classifier):
        self.loader = loader
        self.descriptor = descriptor
        self.classifier = classifier
        self.X = None
        self.y = None
        self.classes_ = None

    def _check_ldc(self):
        if not hasattr(self.loader, '__call__'):
            raise TypeError('loader must be a function which accepts path and returns ndarray')
        if not hasattr(self.descriptor, '__call__'):
            raise TypeError('descriptor must be a function which accepts ndarray and returns ndarray')
        if not (hasattr(self.classifier, 'fit') and hasattr(self.classifier, 'predict_proba')):
            raise TypeError('classifier must implement methods: fit, predict_proba')

    @staticmethod
    def _path_parse(path):
        if isinstance(path, str) and os.path.isdir(path):
            filelist = os.listdir(path)
            return [os.path.join(path, _) for _ in filelist]
        elif hasattr(path, '__iter__'):
            return list(path)
        else:
            raise TypeError('provided path neither a directory nor an iterable')


class Detector2D(_Detector):
    """
    2D object detector.
    Gaussian pyramid image is used to perform multi-scale object detection.
    
    Parameters
    ----------
    loader : function
        A user defined function to load images. It should accept string (path)
        and return ndarray with 2 or 3 dimensions.
    
    descriptor : function
        A user defined function to preprocess image patches before feeding it
        to a classifier. It should accept ndarray and return ndarray.
    
    classifier : object
        This is assumed to implement the scikit-learn classifier interface.
        Either classifier needs to provide a 'fit' and 'predict_proba' methods.
    
    frame : (int, int)
        The size of image patches to sample.
    
    frame_step : float or (int, int), default=0.25
        The step of frame moving. If float, frame_step is frame*frame_step.
    
    scale_step : float, default=0.75
        Downscale step in gaussian pyramid image.
    
    max_scale_steps : int, default=100
        Maximum number of downscale steps.
    
    Attributes
    ----------
    loader : function
    
    descriptor : function
    
    classifier : object
    
    X : ndarray
    
    y : ndarray
    
    \\classes_ : list of strings
    """

    def __init__(self, loader, descriptor, classifier, frame, frame_step=0.25, scale_step=0.75, max_scale_steps=100):
        super(Detector2D, self).__init__(loader, descriptor, classifier)
        self.frame = frame
        self.frame_step = frame_step
        self.scale_step = scale_step
        self.max_scale_steps = max_scale_steps

    def __repr__(self):
        return '''Detector2D(loader=%s,
            descriptor=%s,
            classifier=%s,
            frame=%s,
            frame_step=%s,
            scale_step=%s,
            max_scale_steps=%s)''' % (
            self.loader, self.descriptor, self.classifier, self.frame, self.frame_step, self.scale_step,
            self.max_scale_steps)

    def _check_format_params(self):
        try:
            if len(self.frame) != 2:
                raise TypeError
            self._frame = (int(self.frame[0]), int(self.frame[1]))
        except TypeError:
            raise TypeError('frame must be a tuple with two int elements')
        try:
            self._frame_step = (int(self.frame[0] * self.frame_step), int(self.frame[1] * self.frame_step))
            if self.frame_step <= 0:
                raise TypeError
        except TypeError:
            try:
                if len(self.frame_step) != 2:
                    raise TypeError
                self._frame_step = (int(self.frame_step[0]), int(self.frame_step[1]))
            except TypeError:
                raise TypeError('frame_step must be a float or a tuple with two int elements')
        try:
            self._scale_step = float(self.scale_step)
        except TypeError:
            raise TypeError('scale_step must be a float in range (0,1)')
        if not (0.0 < self.scale_step < 1.0):
            raise ValueError('scale_step must be a float in range (0,1)')
        try:
            self._max_scale_steps = int(self.max_scale_steps)
        except TypeError:
            raise TypeError('max_scale_steps must be an int greater than 0')
        if self.max_scale_steps <= 0:
            raise ValueError('max_scale_steps must be an int greater than 0')

    @staticmethod
    def _check_loader2d_output(img):
        if isinstance(img, numpy.ndarray) and (img.ndim == 2 or img.ndim == 3):
            return
        else:
            raise TypeError('Loader must return ndarray with 2 or 3 dimensions.')

    def _load(self, trg, copy=True):
        if isinstance(trg, numpy.ndarray):
            if copy is True:
                img = trg.copy()
            else:
                img = trg
        else:
            img = self.loader(trg)
        self._check_loader2d_output(img)
        return img

    def _check_frame2d(self, img, how):
        if how == 'equal':
            if img.shape[0] == self._frame[0] and img.shape[1] == self._frame[1]:
                return
            else:
                raise ValueError('image dimensions must be equal to frame size')
        elif how == 'equal_greater':
            if img.shape[0] >= self._frame[0] and img.shape[1] >= self._frame[1]:
                return
            else:
                raise ValueError('image dimensions must greater than or equal to frame size')

    def _crop2d(self, img):
        offsetd0 = numpy.random.randint(0, img.shape[0] - self._frame[0] + 1)
        offsetd1 = numpy.random.randint(0, img.shape[1] - self._frame[1] + 1)
        return img[offsetd0:offsetd0 + self._frame[0], offsetd1:offsetd1 + self._frame[1]]

    def _add_img_data(self, p, y, y_list, X_list, augmentation, augmentation_factor, crop, horizontal_flip,
                      vertical_flip, castom_augmentation_func):
        img = self._load(p)
        if augmentation is True:
            if crop is False:
                self._check_frame2d(img, 'equal')
            elif crop is True:
                self._check_frame2d(img, 'equal_greater')
            for i in range(augmentation_factor):
                if crop is True:
                    img = self._crop2d(img)
                if horizontal_flip is True and numpy.random.randint(2) == 1:
                    img = img[:, ::-1]
                if vertical_flip is True and numpy.random.randint(2) == 1:
                    img = img[::-1]
                if castom_augmentation_func is not None:
                    if not hasattr(castom_augmentation_func, '__call__'):
                        raise TypeError(
                            'castom_augmentation_func must be a function which accepts ndarray and returns ndarray')
                    img = castom_augmentation_func(img)
                desc = self.descriptor(img)
                X_list.append(desc)
                y_list.append(y)
        elif augmentation is False:
            self._check_frame2d(img, 'equal')
            desc = self.descriptor(img)
            X_list.append(desc)
            y_list.append(y)

    def fit(self, path, store_data=True, augmentation=False, augmentation_factor=1, crop=False, horizontal_flip=False,
            vertical_flip=False, custom_augmentation_func=None):
        """
        Fit Detector2D.classifier with passed data.
        If crop is False images should be the same size equal to frame size,
        otherwise images could be equal or greater than frame size.
        
        Parameters
        ----------
        path : string, list of lists of strings or dict of lists of strings
            If string, a path to folder with folders with images. Otherwise,
            strings are paths to individual images. If dict, key is a class name.

        store_data : bool, default=True
            Whether to store the data used to train classifier.
        
        augmentation : bool, default=False
            Whether to use augmentation.
        
        augmentation_factor : int, default=1
            How much samples extract from one image. Useful only when
            augmentation is True.
        
        crop : bool, default=False
            Whether to use crop.
            Useful only when augmentation is True.
        
        horizontal_flip : bool, default=False
            Whether to perform horizontal flip with probability 50%.
            Useful only when augmentation is True.
        
        vertical_flip : bool, default=False
            Whether to perform vertical flip with probability 50%.
            Useful only when augmentation is True.
        
        custom_augmentation_func : function, default=None
            A user defined function to modify images. It should accept 
            ndarray and return ndarray. Useful only when augmentation
            is True.
        
        Returns
        -------
        Detector2D : an instance of self
        """
        self._check_ldc()
        self._check_format_params()
        if isinstance(path, str) and os.path.isdir(path):
            self.classes_ = os.listdir(path)
            cpaths = [os.path.join(path, _) for _ in self.classes_]
        elif isinstance(path, list):
            self.classes_ = [str(_) for _ in range(len(path))]
            cpaths = path
        elif isinstance(path, dict):
            self.classes_ = []
            cpaths = []
            for k in path:
                self.classes_.append(str(k))
                cpaths.append(path[k])
        else:
            raise TypeError('incorrect path')
        y_list = []
        X_list = []
        for i in range(len(self.classes_)):
            path_c_list = self._path_parse(cpaths[i])
            for p in path_c_list:
                self._add_img_data(p, i, y_list, X_list, augmentation, augmentation_factor, crop, horizontal_flip,
                                   vertical_flip, custom_augmentation_func)
        y = numpy.array(y_list)
        X = numpy.stack(X_list)
        if store_data is True:
            self.y = y
            self.X = X
        self.classifier.fit(X, y)
        return self

    def _downscale_img(self, img):
        if img.ndim == 2:
            gbcimg = scipy.ndimage.filters.gaussian_filter(img, 2.0 / (self._scale_step * 6.0), order=0, mode='nearest')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                outimg = scipy.ndimage.zoom(gbcimg, self._scale_step, order=1, mode='nearest')
            return outimg
        elif img.ndim == 3:
            l = []
            for i in range(img.shape[2]):
                gbcimg = scipy.ndimage.filters.gaussian_filter(img[:, :, i], 2.0 / (self._scale_step * 6.0), order=0,
                                                               mode='nearest')
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    l.append(scipy.ndimage.zoom(gbcimg, self._scale_step, order=1, mode='nearest'))
            return numpy.stack(l, axis=2)
        else:
            raise ValueError('image has more than 3 dimensions')

    def detect(self, target, classes=None, threshold=0.5, post_processing='NMS', overlap_threshold=0.3):
        """
        Detect objects on an image using Detector2D.classifier.

        Parameters
        ----------
        target : string or ndarray
            Image to detect objects. If string, path to the image.
            If ndarray, image itself.
        
        classes : string or list of strings or set of strings, default=None
            Which object classes need to detect. If None, all classes would be 
            detected.
        
        threshold : float, default=0.5
            The minimum probability returned by classifier to recognize object.
        
        post_processing : float, default='NMS'
            Which post processing apply to the recognized objects. Options: 'None',
            'NMS'.
        
        overlap_threshold : float, default=0.3
            A parameter for post processing. 0.0 - frames have no overlapping 
            area, 1.0 frames are identical.
        
        Returns
        -------
        Mark2D : Mark2D object with frames of recognized objects
        """
        self._check_ldc()
        self._check_format_params()
        if isinstance(classes, str):
            classes = [classes]
            classes = set(classes)
        elif isinstance(classes, list):
            classes = set(classes)
        elif isinstance(classes, set):
            pass
        elif classes is None:
            classes = set(self.classes_)
        else:
            raise ValueError('classes must be str or set of strings or list of strings or None')
        for cl in classes:
            if cl not in self.classes_:
                raise ValueError("detector don't know class: %s " % cl)
        img = self._load(target)
        frame_list = []
        c = 0
        while True:
            fscale = self._scale_step ** -c
            c += 1
            if img.shape[0] <= self._frame[0] or img.shape[1] <= self._frame[1] or c > self._max_scale_steps:
                break
            for d0 in range(0, img.shape[0] - self._frame[0] + 1, self._frame_step[0]):
                for d1 in range(0, img.shape[1] - self._frame[1] + 1, self._frame_step[1]):
                    Xt = self.descriptor(img[d0:d0 + self._frame[0], d1:d1 + self._frame[1]])[numpy.newaxis, :]
                    probas = self.classifier.predict_proba(Xt)[0]
                    for p, cl in zip(probas, self.classes_):
                        if p >= threshold and cl in classes:
                            frame_list.append((int(d0 * fscale), int(d1 * fscale), int(self._frame[0] * fscale),
                                               int(self._frame[1] * fscale), float(p), str(cl)))
            img = self._downscale_img(img)
        mo = Mark2D(frame_list)
        if post_processing == 'NMS':
            mo = mo.apply_nms(overlap_threshold)
        return mo

    def detect_crop(self, target, classes, threshold=0.5, post_processing='NMS', overlap_threshold=0.3):
        """
        Detect objects on an image using Detector2D.classifier and crop recognized object images.

        Parameters
        ----------
        target : string or ndarray
            Image to detect objects. If string, path to the image.
            If ndarray, image itself.

        classes : string or list of strings or set of strings
            Which object classes need to detect.

        threshold : float, default=0.5
            The minimum probability returned by classifier to recognize object.

        post_processing : float, default='NMS'
            Which post processing apply to the recognized objects. Options: 'None',
            'NMS'.

        overlap_threshold : float, default=0.3
            A parameter for post processing. 0.0 - frames have no overlapping
            area, 1.0 frames are identical.

         Returns
         -------
         list of ndarray : list of cropped images
         """
        img = self._load(target, copy=False)
        mo = self.detect(img, classes, threshold, post_processing, overlap_threshold)
        frame_list = mo.get_list_of_frames()
        img_list = []
        for f in frame_list:
            img_list.append(img[f[0]:f[0] + f[2], f[1]:f[1] + f[3]])
        return img_list

    @staticmethod
    def _drawrect(img, f, color):
        img[f[0]:f[0] + f[2], f[1]] = color
        img[f[0]:f[0] + f[2], f[1] + f[3] - 1] = color
        img[f[0], f[1]:f[1] + f[3]] = color
        img[f[0] + f[2] - 1, f[1]:f[1] + f[3]] = color

    def detect_mark(self, target, classes, threshold=0.5, post_processing='NMS', overlap_threshold=0.3, color=0):
        """
        Detect objects on an image using Detector2D.classifier and paint frames around objects.

        Parameters
        ----------
        target : string or ndarray
            Image to detect objects. If string, path to the image.
            If ndarray, image itself.

        classes : string or list of strings or set of strings
            Which object classes need to detect. If None, all classes would be
            detected.

        threshold : float, default=0.5
            The minimum probability returned by classifier to recognize object.

        post_processing : float, default='NMS'
            Which post processing apply to the recognized objects. Options: 'None',
            'NMS'.

        overlap_threshold : float, default=0.3
            A parameter for post processing. 0.0 - frames have no overlapping
            area, 1.0 frames are identical.

        color : float or int or array like , default=0
            A color to paint frames on image. If array like, len(color) must
            be equal to the number of image channels.

        Returns
        -------
        ndarray : original image with painted frames
        """
        img = self._load(target)
        mo = self.detect(img, classes, threshold, post_processing, overlap_threshold)
        frame_list = mo.get_list_of_frames()
        for f in frame_list:
            self._drawrect(img, f, color)
        return img

    @staticmethod
    def _get_targets(targets):
        if isinstance(targets, str):
            fls = os.listdir(targets)
            targets2 = [os.path.join(targets, _) for _ in fls]
        elif isinstance(targets, list):
            targets2 = targets
        elif isinstance(targets, set):
            targets2 = list(targets)
        else:
            raise ValueError('targets must be str or list of strs or list of ndarray')
        return targets2

    def batch_detect(self, targets, classes=None, threshold=0.5, post_processing='NMS', overlap_threshold=0.3):
        """
        Detect objects on images using Detector2D.classifier.

        Parameters
        ----------
        targets : list of strings or list of ndarray
            List of images to detect objects. If string in list, path to the image.
            If ndarray in list, images itself.

        classes : string or list of strings or set of strings, default=None
            Which object classes need to detect. If None, all classes would be
            detected.

        threshold : float, default=0.5
            The minimum probability returned by classifier to recognize object.

        post_processing : float, default='NMS'
            Which post processing apply to the recognized objects. Options: 'None',
            'NMS'.

        overlap_threshold : float, default=0.3
            A parameter for post processing. 0.0 - frames have no overlapping
            area, 1.0 frames are identical.

        Returns
        -------
        list of Mark2D : list of Mark2D objects with frames of recognized objects
        """
        targets = self._get_targets(targets)
        mos = []
        for t in targets:
            mos.append(self.detect(t, classes, threshold, post_processing, overlap_threshold))
        return mos

    def batch_detect_crop(self, targets, classes, threshold=0.5, post_processing='NMS', overlap_threshold=0.3,
                          flatten=False):
        """
        Detect objects on images using Detector2D.classifier and crop recognized object images.

        Parameters
        ----------
        targets : list of strings or list of ndarray
            List of images to detect objects. If string in list, path to the image.
            If ndarray in list, images itself.

        classes : string or list of strings or set of strings
            Which object classes need to detect.

        threshold : float, default=0.5
            The minimum probability returned by classifier to recognize object.

        post_processing : float, default='NMS'
            Which post processing apply to the recognized objects. Options: 'None',
            'NMS'.

        overlap_threshold : float, default=0.3
            A parameter for post processing. 0.0 - frames have no overlapping
            area, 1.0 frames are identical.

        flatten : bool, default=False
            Whether to flatten list of images.

        Returns
        -------
        list : list of lists of ndarray or list of ndarray
        """
        targets = self._get_targets(targets)
        if flatten is False:
            img_lists = []
            for t in targets:
                img_lists.append(self.detect_crop(t, classes, threshold, post_processing, overlap_threshold))
            return img_lists
        else:
            img_list = []
            for t in targets:
                img_list += self.detect_crop(t, classes, threshold, post_processing, overlap_threshold)
            return img_list

    def batch_detect_mark(self, targets, classes, threshold=0.5, post_processing='NMS', overlap_threshold=0.3, color=0):
        """
        Detect objects on images using Detector2D.classifier and paint frames around objects.

        Parameters
        ----------
        targets : list of strings or list of ndarray
            List of images to detect objects. If string in list, path to the image.
            If ndarray in list, images itself.

        classes : string or list of strings or set of strings
            Which object classes need to detect.

        threshold : float, default=0.5
            The minimum probability returned by classifier to recognize object.

        post_processing : float, default='NMS'
            Which post processing apply to the recognized objects. Options: 'None',
            'NMS'.

        overlap_threshold : float, default=0.3
            A parameter for post processing. 0.0 - frames have no overlapping
            area, 1.0 frames are identical.

        color : float or int or array like , default=0
            A color to paint frames on image. If array like, len(color) must
            be equal to the number of image channels.

        Returns
        -------
        list of ndarray : original images with painted frames
        """
        targets = self._get_targets(targets)
        img_list = []
        for t in targets:
            img_list.append(self.detect_mark(t, classes, threshold, post_processing, overlap_threshold, color))
        return img_list
