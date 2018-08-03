"""*********************************************************************************************************************
 * Name: setup.py
 * Author: Matin Raayai Ardakani
 * Email: raayai.matin@gmail.com
 * Setup script for the package em_pre_cuda.
 ********************************************************************************************************************"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
NAME = "em_pre_cuda"
VERSION = "0.0.1"
DESCR = "CUDA implementation of a 3D median filter for PyTorch."
REQUIRES = ['torch']
AUTHOR = "Donglai Wei, Tran Minh Quan, Matin Raayai Ardakani"
EMAIL = "raayai.matin@gmail.com"
LICENSE = "Apache 2.0"
SRC_DIR = "src"
FILES = [SRC_DIR + '/em_pre_cuda.cpp', SRC_DIR + '/em_pre_cuda.h', SRC_DIR + '/em_pre_cuda_kernel.cu', SRC_DIR + '/em_pre_cuda_kernel.h']
EXTENSION = []
EXTENSION += [CUDAExtension(NAME, [SRC_DIR + '/em_pre_cuda.cpp', SRC_DIR + '/em_pre_cuda_kernel.cu'])]


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