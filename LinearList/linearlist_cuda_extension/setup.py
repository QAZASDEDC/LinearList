from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='linearlist_cuda',
    ext_modules=[
        CUDAExtension('linearlist_cuda', [
            'linearlist_cuda.cpp',
            'linearlist_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })