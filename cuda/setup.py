from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).parent

setup(
    name="inference-lab-image-preprocess-cuda",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            "image_preprocess_cuda",
            [str(root / "src" / "image_preprocess.cpp"), str(root / "src" / "image_preprocess_kernel.cu")],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
