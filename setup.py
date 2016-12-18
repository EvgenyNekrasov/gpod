# -*- coding: utf-8 -*-
from __future__ import absolute_import
from setuptools import setup

setup(name='gpod',
      version='0.1.2',
      description='general purpose object detector',
      long_description='general purpose object detector',
      url='https://github.com/EvgenyNekrasov/gpod',
      author='Evgeny Nekrasov',
      author_email='evgeny.nekrasov@phystech.edu',
      license='BSD 3-Clause License',
      classifiers=['Development Status :: 4 - Beta',
                   'Topic :: Software Development :: Libraries',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence',
                   'Topic :: Scientific/Engineering :: Image Recognition',
                   'License :: OSI Approved :: BSD License',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3.5'],
      keywords='image recognition vision detection',
      packages=['gpod'],
      install_requires=['numpy', 'scipy'],
      extras_require={},
      )
