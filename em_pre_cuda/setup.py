from setuptools import setup, Extension, find_packages
from Cython.Distutils import build_ext
import numpy as np
import glob

NAME = "em_pre_cuda"
VERSION = "0.0.6"
DESCR = "C++/CUDA implementation of Deformation Models for Image Recognition"
REQUIRES = ['cython', 'numpy', 'opencv-python', 'torch', 'h5py', 'em_pre_torch_ext']

AUTHOR = "Donglai Wei, Matin Raayai Ardakani"
EMAIL = "weiddoonngglai@gmail.com, raayai.matin@gmail.com"
LICENSE = "Apache 2.0"
SRC_DIR = "em_pre_cuda"
URL = 'https://github.com/donglaiw/EM-preprocess'
PACKAGES = find_packages()

if __name__ == "__main__":
    # python setup.py develop install
    setup(install_requires=REQUIRES,
          packages=PACKAGES,
          zip_safe=False,
          url=URL,
          name=NAME,
          version=VERSION,
          description=DESCR,
          author=AUTHOR,
          author_email=EMAIL,
          license=LICENSE,
          cmdclass={"build_ext": build_ext}
          )
