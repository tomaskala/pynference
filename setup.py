from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="truncated_normal_cpp",
    ext_modules=[
        cpp_extension.CppExtension(
            "truncated_normal_cpp", ["pynference/distributions/truncated_normal.cpp"]
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
