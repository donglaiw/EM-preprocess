from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy as np

NAME = "image distortion model"
VERSION = "0.1"
DESCR = "cython implementation of Deformation Models for Image Recognition"
REQUIRES = ['numpy', 'cython']

AUTHOR = "Donglai Wei"
EMAIL = "weiddoonngglai@gmail.com"
LICENSE = "Apache 2.0"
SRC_DIR = "idm"
PACKAGES = [SRC_DIR]

ext_1 = Extension(SRC_DIR + ".wrapped",
                  [SRC_DIR + "/src/idm.c", SRC_DIR + "/idm_main.pyx"],
                  libraries=[],
                  include_dirs=[np.get_include()])

EXTENSIONS = [ext_1]

if __name__ == "__main__":
    setup(install_requires=REQUIRES,
          packages=PACKAGES,
          zip_safe=False,
          name=NAME,
          version=VERSION,
          description=DESCR,
          author=AUTHOR,
          author_email=EMAIL,
          url=URL,
          license=LICENSE,
          cmdclass={"build_ext": build_ext},
          ext_modules=EXTENSIONS
          )
