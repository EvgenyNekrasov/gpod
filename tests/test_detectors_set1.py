# -*- coding: utf-8 -*-

import pytest
import gpod

import numpy
from sklearn.base import BaseEstimator, ClassifierMixin

data = {}
data['pos1'] = numpy.ones((16, 8))
data['pos2'] = numpy.ones((16, 8))
data['neg1'] = numpy.zeros((16, 8))
data['neg2'] = numpy.zeros((16, 8))
data['neg3'] = numpy.zeros((16, 8))

data['posd1'] = numpy.ones((20, 10))
data['negd1'] = numpy.zeros((20, 10))

img1 = numpy.zeros((100, 100))
img1[10:10 + 20, 10:10 + 10] += 1  # (10,10)
img1[20:20 + 20, 40:40 + 10] += 1  # (20,40)

img2 = numpy.zeros((100, 100, 2))
img2[10:10 + 20, 10:10 + 10, 0] += 2  # (10,10)
img2[20:20 + 20, 40:40 + 10, 1] += 2  # (20,40)


def lo1(s):
    return data[s]


def de1(a):
    return a.flatten()


class CL1(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict_proba(self, X):
        return numpy.stack([X.mean(axis=1), numpy.zeros(X.shape[0])]).T[:, ::-1]


class Test_Detector2D:
    def test_create(self):
        d1 = gpod.Detector2D(lo1, de1, CL1(), (16, 8), frame_step=0.5, scale_step=0.5, max_scale_steps=1)
        d1._check_ldc()
        d1._check_format_params()
        assert d1._frame == (16, 8)
        assert d1._frame_step == (8, 4)
        assert d1._max_scale_steps == 1
        d2 = gpod.Detector2D(lo1, de1, CL1(), (16, 8), frame_step=(8, 4), scale_step=0.5, max_scale_steps=4)
        d2._check_ldc()
        d2._check_format_params()
        assert d2._frame == (16, 8)
        assert d2._frame_step == (8, 4)
        assert d2._max_scale_steps == 4
        with pytest.raises(TypeError):
            d3 = gpod.Detector2D(lo1, CL1(), CL1(), (16, 8), frame_step=(8, 4), scale_step=0.5, max_scale_steps=4)
            d3._check_ldc()
            d3._check_format_params()
        with pytest.raises(TypeError):
            d3 = gpod.Detector2D(lo1, de1, de1, (16, 8), frame_step=(8, 4), scale_step=0.5, max_scale_steps=4)
            d3._check_ldc()
            d3._check_format_params()
        with pytest.raises(TypeError):
            d3 = gpod.Detector2D(CL1(), de1, CL1(), (16, 8), frame_step=(8, 4), scale_step=0.5, max_scale_steps=4)
            d3._check_ldc()
            d3._check_format_params()
        with pytest.raises(TypeError):
            d3 = gpod.Detector2D(lo1, de1, CL1(), (16,), frame_step=(8, 4), scale_step=0.5, max_scale_steps=4)
            d3._check_ldc()
            d3._check_format_params()
        with pytest.raises(TypeError):
            d3 = gpod.Detector2D(lo1, de1, CL1(), (16, 8, 2), frame_step=(8, 4), scale_step=0.5, max_scale_steps=4)
            d3._check_ldc()
            d3._check_format_params()
        with pytest.raises(TypeError):
            d3 = gpod.Detector2D(lo1, de1, CL1(), (16, 8), frame_step=(8,), scale_step=0.5, max_scale_steps=4)
            d3._check_ldc()
            d3._check_format_params()
        with pytest.raises(TypeError):
            d3 = gpod.Detector2D(lo1, de1, CL1(), (16, 8), frame_step=-1, scale_step=0.5, max_scale_steps=4)
            d3._check_ldc()
            d3._check_format_params()
        with pytest.raises(TypeError):
            d3 = gpod.Detector2D(lo1, de1, CL1(), -1, frame_step=(8, 4), scale_step=0.5, max_scale_steps=4)
            d3._check_ldc()
            d3._check_format_params()
        with pytest.raises(ValueError):
            d3 = gpod.Detector2D(lo1, de1, CL1(), (16, 8), frame_step=(8, 4), scale_step=1.5, max_scale_steps=4)
            d3._check_ldc()
            d3._check_format_params()
        with pytest.raises(ValueError):
            d3 = gpod.Detector2D(lo1, de1, CL1(), (16, 8), frame_step=(8, 4), scale_step=-0.5, max_scale_steps=4)
            d3._check_ldc()
            d3._check_format_params()
        with pytest.raises(ValueError):
            d3 = gpod.Detector2D(lo1, de1, CL1(), (16, 8), frame_step=(8, 4), scale_step=0.0, max_scale_steps=4)
            d3._check_ldc()
            d3._check_format_params()
        with pytest.raises(ValueError):
            d3 = gpod.Detector2D(lo1, de1, CL1(), (16, 8), frame_step=(8, 4), scale_step=0.5, max_scale_steps=0)
            d3._check_ldc()
            d3._check_format_params()
        with pytest.raises(ValueError):
            d3 = gpod.Detector2D(lo1, de1, CL1(), (16, 8), frame_step=(8, 4), scale_step=0.5, max_scale_steps=-6)
            d3._check_ldc()
            d3._check_format_params()

    def test_fit(self):
        d1 = gpod.Detector2D(lo1, de1, CL1(), (16, 8), frame_step=0.5, scale_step=0.5, max_scale_steps=1)
        path1 = [[data['pos1'], data['pos2']], [data['neg1'], data['neg2'], data['neg3']]]
        d1.fit(path1, store_data=True)
        assert d1.X.shape == (5, 128)
        assert d1.y.shape == (5,)
        assert d1.classes_ == ['0', '1']
        d2 = gpod.Detector2D(lo1, de1, CL1(), (16, 8), frame_step=0.5, scale_step=0.5, max_scale_steps=1)
        path2 = [['pos1', 'pos2'], ['neg1', 'neg2', 'neg3']]
        d2.fit(path2, store_data=True)
        assert d2.X.shape == (5, 128)
        assert d2.y.shape == (5,)
        assert d2.classes_ == ['0', '1']
        d3 = gpod.Detector2D(lo1, de1, CL1(), (16, 8), frame_step=0.5, scale_step=0.5, max_scale_steps=1)
        path3 = {'pos': ['pos1', 'pos2'], 'neg': ['neg1', 'neg2', 'neg3']}
        d3.fit(path3, store_data=True)
        assert d3.X.shape == (5, 128)
        assert d3.y.shape == (5,)
        assert d3.classes_ == ['pos', 'neg'] or d3.classes_ == ['neg', 'pos']
        d4 = gpod.Detector2D(lo1, de1, CL1(), (16, 8), frame_step=0.5, scale_step=0.5, max_scale_steps=1)
        with pytest.raises(TypeError):
            d4.fit(1)
        d4 = gpod.Detector2D(lo1, de1, CL1(), (17, 4), frame_step=0.5, scale_step=0.5, max_scale_steps=1)
        path4 = {'pos': ['pos1', 'pos2'], 'neg': ['neg1', 'neg2', 'neg3']}
        with pytest.raises(ValueError):
            d4.fit(path4)

    def test_detect(self):
        d1 = gpod.Detector2D(lo1, de1, CL1(), (20, 10), frame_step=(10, 10), scale_step=0.5, max_scale_steps=1)
        d1.fit([['posd1'], ['negd1']])
        mo1 = d1.detect(img1, classes=None, threshold=0.0, post_processing='None', overlap_threshold=0.0)
        assert isinstance(mo1, gpod.Mark2D)
        assert len(mo1) == 180
        assert len(mo1['0']) == 90
        assert len(mo1[0.4:]) == 6
        assert len(mo1[0.6:]) == 2
        assert len(mo1[0.4:].apply_nms(0.3)) == 2
        assert len(mo1[0.4:]['1'].apply_nms(0.3)) == 2
        assert len(mo1['1'][0.4:].apply_nms(0.3)) == 2
        lof = mo1['1'][0.4:].apply_nms(0.3).get_list_of_frames()
        assert lof[0] == (10, 10, 20, 10, 1.0, '1')
        assert lof[1] == (20, 40, 20, 10, 1.0, '1')
        d2 = gpod.Detector2D(lo1, de1, CL1(), (20, 10), frame_step=(10, 10), scale_step=0.5, max_scale_steps=2)
        d2.fit([['posd1'], ['negd1']])
        mo2 = d2.detect(img1, classes=None, threshold=0.0, post_processing='None', overlap_threshold=0.0)
        assert isinstance(mo2, gpod.Mark2D)
        assert len(mo2) == 220
        assert len(mo2['0']) == 110
        assert len(mo2[0.10:]) == 6 + 4
        assert len(mo2[0.13:]) == 6 + 3
        assert len(mo2[0.26:]) == 6 + 0
        assert len(mo2[0.4:].apply_nms(0.3)) == 2
        assert len(mo2[0.4:]['1'].apply_nms(0.3)) == 2
        assert len(mo2['1'][0.4:].apply_nms(0.3)) == 2
        lof = mo2['1'][0.4:].apply_nms(0.3).get_list_of_frames()
        assert lof[0] == (10, 10, 20, 10, 1.0, '1')
        assert lof[1] == (20, 40, 20, 10, 1.0, '1')
        d3 = gpod.Detector2D(lo1, de1, CL1(), (20, 10), frame_step=(10, 10), scale_step=0.5, max_scale_steps=3)
        d3.fit([['posd1'], ['negd1']])
        mo3 = d3.detect(img1, classes=None, threshold=0.0, post_processing='None', overlap_threshold=0.0)
        assert len(mo3) == 224
        assert len(mo3['1']) == 112
        assert len(mo3[0.4:].apply_nms(0.3)) == 2
        assert len(mo3[0.4:]['1'].apply_nms(0.3)) == 2
        assert len(mo3['1'][0.4:].apply_nms(0.3)) == 2
        lof = mo3['1'][0.4:].apply_nms(0.3).get_list_of_frames()
        assert lof[0] == (10, 10, 20, 10, 1.0, '1')
        assert lof[1] == (20, 40, 20, 10, 1.0, '1')
        d4 = gpod.Detector2D(lo1, de1, CL1(), (20, 10), frame_step=(10, 10), scale_step=0.5, max_scale_steps=100)
        d4.fit([['posd1'], ['negd1']])
        mo4 = d4.detect(img1, classes=None, threshold=0.0, post_processing='None', overlap_threshold=0.0)
        assert len(mo4) == 224
        assert len(mo4['1']) == 112
        assert len(mo4[0.4:].apply_nms(0.3)) == 2
        assert len(mo4[0.4:]['1'].apply_nms(0.3)) == 2
        assert len(mo4['1'][0.4:].apply_nms(0.3)) == 2
        lof = mo4['1'][0.4:].apply_nms(0.3).get_list_of_frames()
        assert lof[0] == (10, 10, 20, 10, 1.0, '1')
        assert lof[1] == (20, 40, 20, 10, 1.0, '1')
        d5 = gpod.Detector2D(lo1, de1, CL1(), (20, 10), frame_step=(10, 10), scale_step=0.5, max_scale_steps=1)
        d5.fit([['posd1'], ['negd1']])
        mo5 = d5.detect(img1, classes=None, threshold=0.01, post_processing='NMS', overlap_threshold=0.3)
        assert len(mo5) == 2
        lof = mo5.get_list_of_frames()
        assert lof[0] == (10, 10, 20, 10, 1.0, '1')
        assert lof[1] == (20, 40, 20, 10, 1.0, '1')
        d6 = gpod.Detector2D(lo1, de1, CL1(), (20, 10), frame_step=(10, 10), scale_step=0.5, max_scale_steps=1)
        d6.fit([['posd1'], ['negd1']])
        mo6 = d6.detect(img1, classes='1', threshold=0.6, post_processing='None', overlap_threshold=0.0)
        assert len(mo6) == 2
        lof = mo6.get_list_of_frames()
        assert lof[0] == (10, 10, 20, 10, 1.0, '1')
        assert lof[1] == (20, 40, 20, 10, 1.0, '1')
        d7 = gpod.Detector2D(lo1, de1, CL1(), (20, 10), frame_step=(10, 10), scale_step=0.5, max_scale_steps=2)
        d7.fit([['posd1'], ['negd1']])
        mo7 = d7.detect(img2, classes=None, threshold=0.0, post_processing='None', overlap_threshold=0.0)
        assert isinstance(mo7, gpod.Mark2D)
        assert len(mo7) == 220
        assert len(mo7['0']) == 110
        assert len(mo7[0.10:]) == 6 + 4
        assert len(mo7[0.13:]) == 6 + 3
        assert len(mo7[0.26:]) == 6 + 0
        assert len(mo7[0.4:].apply_nms(0.3)) == 2
        assert len(mo7[0.4:]['1'].apply_nms(0.3)) == 2
        assert len(mo7['1'][0.4:].apply_nms(0.3)) == 2
        lof = mo7['1'][0.4:].apply_nms(0.3).get_list_of_frames()
        assert lof[0] == (10, 10, 20, 10, 1.0, '1')
        assert lof[1] == (20, 40, 20, 10, 1.0, '1')
        mo7 = d7.detect(img2, classes='1', threshold=0.0, post_processing='None', overlap_threshold=0.0)
        mo7 = d7.detect(img2, classes=['1'], threshold=0.0, post_processing='None', overlap_threshold=0.0)
        mo7 = d7.detect(img2, classes=set(['1']), threshold=0.0, post_processing='None', overlap_threshold=0.0)
        with pytest.raises(ValueError):
            mo7 = d7.detect(img2, classes='pos', threshold=0.0, post_processing='None', overlap_threshold=0.0)
        with pytest.raises(ValueError):
            mo7 = d7.detect(img2, classes=['pos'], threshold=0.0, post_processing='None', overlap_threshold=0.0)
        with pytest.raises(ValueError):
            mo7 = d7.detect(img2, classes=set(['pos']), threshold=0.0, post_processing='None', overlap_threshold=0.0)
        with pytest.raises(ValueError):
            mo7 = d7.detect(img2, classes=1, threshold=0.0, post_processing='None', overlap_threshold=0.0)

    def test_detect_crop(self):
        d1 = gpod.Detector2D(lo1, de1, CL1(), (20, 10), frame_step=(10, 10), scale_step=0.5, max_scale_steps=1)
        d1.fit([['posd1'], ['negd1']])
        il1 = d1.detect_crop(img1, classes='1', threshold=0.6, post_processing='None', overlap_threshold=0.0)
        assert len(il1) == 2
        assert il1[0].shape == (20, 10)
        assert il1[1].shape == (20, 10)
        assert il1[0].sum() == 200
        assert il1[1].sum() == 200
        d2 = gpod.Detector2D(lo1, de1, CL1(), (20, 10), frame_step=(10, 10), scale_step=0.5, max_scale_steps=1)
        d2.fit([['posd1'], ['negd1']])
        il2 = d2.detect_crop(img2, classes='1', threshold=0.6, post_processing='None', overlap_threshold=0.0)
        assert len(il2) == 2
        assert il2[0].shape == (20, 10, 2)
        assert il2[1].shape == (20, 10, 2)
        assert il2[0].sum() == 400
        assert il2[1].sum() == 400

    def test_detect_mark(self):
        d1 = gpod.Detector2D(lo1, de1, CL1(), (20, 10), frame_step=(10, 10), scale_step=0.5, max_scale_steps=1)
        d1.fit([['posd1'], ['negd1']])
        im1 = d1.detect_mark(img1, classes='1', threshold=0.6, post_processing='None', overlap_threshold=0.0)
        assert isinstance(im1, numpy.ndarray)
        assert im1.shape == (100, 100)
        assert im1.sum() == 200 + 200 - 56 - 56
        d2 = gpod.Detector2D(lo1, de1, CL1(), (20, 10), frame_step=(10, 10), scale_step=0.5, max_scale_steps=1)
        d2.fit([['posd1'], ['negd1']])
        im2 = d2.detect_mark(img1, classes='1', threshold=0.6, post_processing='None', overlap_threshold=0.0, color=3)
        assert isinstance(im2, numpy.ndarray)
        assert im2.shape == (100, 100)
        assert im2.sum() == 200 + 200 + 56 * 2 + 56 * 2
        d3 = gpod.Detector2D(lo1, de1, CL1(), (20, 10), frame_step=(10, 10), scale_step=0.5, max_scale_steps=1)
        d3.fit([['posd1'], ['negd1']])
        im3 = d3.detect_mark(img2, classes='1', threshold=0.6, post_processing='None', overlap_threshold=0.0, color=(3, 4))
        assert isinstance(im3, numpy.ndarray)
        assert im3.shape == (100, 100, 2)
        assert im3.sum() == 400 + 400 + 56 * 5 + 56 * 5

    def test_batch_detect(self):
        d1 = gpod.Detector2D(lo1, de1, CL1(), (20, 10), frame_step=(10, 10), scale_step=0.5, max_scale_steps=1)
        d1.fit([['posd1'], ['negd1']])
        bimgs = [img1, img1, img1]
        lmo1 = d1.batch_detect(bimgs, classes=None, threshold=0.0, post_processing='None', overlap_threshold=0.0)
        assert isinstance(lmo1, list)
        assert len(lmo1) == 3
        assert isinstance(lmo1[0], gpod.Mark2D)

    def test_batch_detect_crop(self):
        d1 = gpod.Detector2D(lo1, de1, CL1(), (20, 10), frame_step=(10, 10), scale_step=0.5, max_scale_steps=1)
        d1.fit([['posd1'], ['negd1']])
        bimgs = [img1, img1, img1]
        lmo1 = d1.batch_detect_crop(bimgs, classes=None, threshold=0.0, post_processing='None', overlap_threshold=0.0,
                                    flatten=False)
        assert isinstance(lmo1, list)
        assert len(lmo1) == 3
        assert isinstance(lmo1[0], list)
        assert isinstance(lmo1[0][0], numpy.ndarray)
        d2 = gpod.Detector2D(lo1, de1, CL1(), (20, 10), frame_step=(10, 10), scale_step=0.5, max_scale_steps=1)
        d2.fit([['posd1'], ['negd1']])
        bimgs = [img1, img1, img1]
        lmo2 = d2.batch_detect_crop(bimgs, classes=None, threshold=0.6, post_processing='None', overlap_threshold=0.0,
                                    flatten=True)
        assert isinstance(lmo2, list)
        assert len(lmo2) == 6
        assert isinstance(lmo2[0], numpy.ndarray)
        assert lmo2[0].shape == (20, 10)

    def test_batch_detect_mark(self):
        d1 = gpod.Detector2D(lo1, de1, CL1(), (20, 10), frame_step=(10, 10), scale_step=0.5, max_scale_steps=1)
        d1.fit([['posd1'], ['negd1']])
        bimgs = [img1, img1, img1]
        lmo1 = d1.batch_detect_mark(bimgs, classes=None, threshold=0.0, post_processing='None', overlap_threshold=0.0)
        assert isinstance(lmo1, list)
        assert len(lmo1) == 3
        assert isinstance(lmo1[0], numpy.ndarray)
        assert lmo1[0].shape == (100, 100)
