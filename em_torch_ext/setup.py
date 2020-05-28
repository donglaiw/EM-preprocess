"""*********************************************************************************************************************
 * Name: setup.py
 * Author: Matin Raayai Ardakani
 * Email: matinraayai@seas.harvard.edu
 * Setup script for the package em_torch_ext.
 ********************************************************************************************************************"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
NAME = "em_torch_ext"
VERSION = "0.1"
DESCR = "Pytorch C++/CUDA extensions needed for other EM Python packages."
REQUIRES = ['torch']
AUTHOR = "Donglai Wei, Matin Raayai Ardakani"
EMAIL = "weiddoonngglai@gmail.com, matinraayai@seas.harvard.edu"
LICENSE = "Apache 2.0"
SRC_DIR = "em_torch_ext"
EXTENSION = []
EXTENSION += [CUDAExtension(NAME, [SRC_DIR + '/TorchExtension.cpp', SRC_DIR + '/TorchExtensionKernel.cu'])]


setup(
    install_requires=REQUIRES,
    zip_safe=False,
    name=NAME,
    version=VERSION,
    description=DESCR,
    author=AUTHOR,
    author_email=EMAIL,
    license=LICENSE,
    ext_modules=EXTENSION,
    cmdclass={
        'build_ext': BuildExtension
    })
