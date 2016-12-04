# -*- coding: utf-8 -*-

import pytest
import gpod


class Test_Mark2D:
    def test_create(self):
        lof1 = [(10, 10, 4, 8, 0.3, 'pos'),
                (10, 20, 4, 8, 0.5, 'pos'),
                (10, 30, 4, 8, 0.2, 'neg'),
                ]
        m1 = gpod.Mark2D(lof1)
        assert m1.shape == (3, 2)
        assert len(m1.classes) == 2
        assert ('pos' in m1) is True
        assert ('neg' in m1) is True
        assert ('smt' in m1) is False
        with pytest.raises(ValueError):
            m2 = gpod.Mark2D(1)
            m3 = gpod.Mark2D('txt')

    def test_slice(self):
        lof1 = [(10, 10, 4, 8, 0.6, 'pos'),
                (10, 20, 4, 8, 0.7, 'pos'),
                (10, 30, 4, 8, 0.4, 'neg'),
                (10, 40, 4, 8, 0.3, 'neg'),
                (10, 50, 4, 8, 0.2, 'neg'),
                (10, 60, 4, 8, 0.1, 'neg'),
                (10, 70, 4, 8, 0.9, 'pos'),
                ]
        m1 = gpod.Mark2D(lof1)
        m2 = m1[set(['pos', 'neg'])]
        assert m2.shape == (7, 2)
        assert len(m2.classes) == 2
        assert ('pos' in m2) is True
        assert ('neg' in m2) is True
        m3 = m1[set(['pos'])]
        assert m3.shape == (3, 1)
        assert len(m3.classes) == 1
        assert ('pos' in m3) is True
        assert ('neg' in m3) is False
        m4 = m1['neg']
        assert m4.shape == (4, 1)
        assert len(m4.classes) == 1
        assert ('pos' in m4) is False
        assert ('neg' in m4) is True
        m5 = m1[0.35:]
        assert m5.shape == (4, 2)
        assert len(m5.classes) == 2
        assert ('pos' in m5) is True
        assert ('neg' in m5) is True
        m6 = m1[0.25:0.65]
        assert m6.shape == (3, 2)
        assert len(m6.classes) == 2
        assert ('pos' in m6) is True
        assert ('neg' in m6) is True
        m7 = m1[:0.25]
        assert m7.shape == (2, 1)
        assert len(m7.classes) == 1
        assert ('pos' in m7) is False
        assert ('neg' in m7) is True

    def test_get_list_of_frames(self):
        lof1 = [(10, 10, 4, 8, 0.3, 'pos'),
                (10, 20, 4, 8, 0.5, 'pos'),
                (10, 30, 4, 8, 0.2, 'neg'),
                (40, 30, 4, 8, 0.1, 'neg')
                ]
        m1 = gpod.Mark2D(lof1)
        lof2 = m1.get_list_of_frames()
        assert len(lof1) == len(lof2)
        for i in range(len(lof1)):
            assert lof1[i] == lof2[i]

    def test_apply_nms(self):
        lof1 = [(10, 10, 10, 20, 0.6, 'pos'),
                (11, 12, 10, 20, 0.7, 'pos'),
                (10, 30, 10, 20, 0.4, 'neg'),
                (10, 40, 10, 20, 0.5, 'neg'),
                (10, 50, 10, 20, 0.2, 'neg'),
                (11, 51, 10, 20, 0.5, 'pos'),
                ]
        m1 = gpod.Mark2D(lof1)
        m2 = m1.apply_nms(0.65)
        assert m2.shape == (5, 2)
        assert ('pos' in m2) is True
        assert ('neg' in m2) is True
        m3 = m1.apply_nms(0.3)
        assert m3.shape == (3, 2)
        assert ('pos' in m3) is True
        assert ('neg' in m3) is True

    def test_frame_overlap(self):
        f1 = (50, 50, 20, 20)
        f2 = (60, 60, 20, 20)
        f3 = (40, 60, 20, 20)
        f4 = (100, 100, 20, 20)
        mo = gpod.Mark2D()
        assert mo._frame_overlap(f1, f4) == 0
        assert mo._frame_overlap(f2, f4) == 0
        assert mo._frame_overlap(f3, f4) == 0
        assert mo._frame_overlap(f4, f1) == 0
        assert mo._frame_overlap(f4, f1) == 0
        assert mo._frame_overlap(f4, f2) == 0
        assert mo._frame_overlap(f4, f3) == 0
        assert mo._frame_overlap(f1, f1) == 1.0
        assert mo._frame_overlap(f2, f2) == 1.0
        assert mo._frame_overlap(f3, f3) == 1.0
        assert mo._frame_overlap(f4, f4) == 1.0
        assert mo._frame_overlap(f1, f2) == 1.0 / 7.0
        assert mo._frame_overlap(f2, f1) == 1.0 / 7.0
        assert mo._frame_overlap(f1, f3) == 1.0 / 7.0
        assert mo._frame_overlap(f3, f1) == 1.0 / 7.0
        assert mo._frame_overlap(f2, f3) == 0.0
