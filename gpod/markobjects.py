# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import numpy


class Mark2D:
    """
    Mark2D object is a container for object frames on image.
    It supports slicing by string or set of strings to extract frames
    of some class or classes. It supports slicing by float range to
    extract frames of probabilities in specified range.

    Parameters
    ----------
    list_of_frames : list
        A list of tuples:
        (x: int, y: int, size_x: int, size_y : int, probability : float, class: string)
    """
    
    def __init__(self, list_of_frames=None):
        if list_of_frames is None:
            list_of_frames = []
        if not isinstance(list_of_frames, list):
            raise ValueError('list_of_frames must be a list of tuples (x, y, size_x, size_y, probability, obj_class)')
        d0 = []
        d1 = []
        sd0 = []
        sd1 = []
        p = []
        nm = []
        for f in list_of_frames:
            d0.append(f[0])
            d1.append(f[1])
            sd0.append(f[2])
            sd1.append(f[3])
            p.append(f[4])
            nm.append(f[5])
        self.d0 = numpy.array(d0, dtype=numpy.int)
        self.d1 = numpy.array(d1, dtype=numpy.int)
        self.sd0 = numpy.array(sd0, dtype=numpy.int)
        self.sd1 = numpy.array(sd1, dtype=numpy.int)
        self.p = numpy.array(p, dtype=numpy.float)
        self.nm = numpy.array(nm, dtype=numpy.str)
        self.classes = numpy.unique(self.nm)
        self.shape = (self.nm.shape[0], self.classes.shape[0])
    
    def __repr__(self):
        return 'Mark2D object with %d frames of %d classes' % self.shape
    
    def __len__(self):
        return self.shape[0]
    
    def __iter__(self):
        return self.classes.__iter__()
    
    def __contains__(self, item):
        return item in self.classes
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            if key.start is not None and key.stop is None:
                mask = self.p >= key.start
            elif key.start is None and key.stop is not None:
                mask = self.p < key.stop
            elif key.start is not None and key.stop is not None:
                mask = (self.p >= key.start) & (self.p < key.stop)
        elif isinstance(key, str) or isinstance(key, numpy.str_):
            mask = self.nm == key
        elif isinstance(key, set):
            mask = numpy.zeros(self.nm.shape, dtype=numpy.bool)
            for s in key:
                mask |= self.nm == s
        else:
            raise KeyError()
        new = Mark2D([])
        new.d0 = self.d0[mask].copy()
        new.d1 = self.d1[mask].copy()
        new.sd0 = self.sd0[mask].copy()
        new.sd1 = self.sd1[mask].copy()
        new.p = self.p[mask].copy()
        new.nm = self.nm[mask].copy()
        new.classes = numpy.unique(new.nm)
        new.shape = (new.nm.shape[0], new.classes.shape[0])
        return new
    
    def get_list_of_frames(self):
        """
        Returns
        -------
        list : list of frames
        """
        lof = []
        for i in range(self.shape[0]):
            lof.append((int(self.d0[i]), int(self.d1[i]), int(self.sd0[i]), int(self.sd1[i]), float(self.p[i]),
                        str(self.nm[i])))
        return lof
    
    @staticmethod
    def _frame_overlap(frame1, frame2):
        d0_overlap = max(0, min(frame1[0] + frame1[2], frame2[0] + frame2[2]) - max(frame1[0], frame2[0]))
        d1_overlap = max(0, min(frame1[1] + frame1[3], frame2[1] + frame2[3]) - max(frame1[1], frame2[1]))
        overlap_area = d0_overlap * d1_overlap
        area_1 = frame1[2] * frame1[3]
        area_2 = frame2[2] * frame2[3]
        total_area = area_1 + area_2 - overlap_area
        return overlap_area / total_area
    
    def _apply_nms_1c(self, frame_list, overlap_threshold):
        if len(frame_list) == 0:
            return []
        frame_list.sort(key=lambda x: x[4])
        new_frame_list = []
        for i in range(0, len(frame_list)):
            for j in range(i + 1, len(frame_list)):
                if self._frame_overlap(frame_list[i], frame_list[j]) > overlap_threshold:
                    break
            else:
                new_frame_list.append(frame_list[i])
        return new_frame_list
    
    def apply_nms(self, overlap_threshold):
        """
        Apply non-maximum suppression (NMS).

        Parameters
        ----------
        overlap_threshold : float
            A parameter for NMS. 0.0 - frames have no overlapping
            area, 1.0 frames are identical.

        Returns
        -------
        Mark2D : Mark2D with filtered frames
        """
        lof = []
        for c in self:
            lof += self._apply_nms_1c(self[c].get_list_of_frames(), overlap_threshold)
        return Mark2D(lof)
