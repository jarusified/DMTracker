import os
import re
import subprocess
import sys
import platform

from distutils.version import LooseVersion
from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir="."):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)',
                                        out.decode()).group(1))

        if cmake_version < '3.16.0':
            raise RuntimeError("CMake >= 3.16.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                cfg.upper(),
                extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''), self.distribution.get_version())

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        FMT_SOURCE_DIR = os.environ["FMT_SOURCE_DIR"]
        GOOGLETEST_SOURCE_DIR = os.environ["GOOGLETEST_SOURCE_DIR"]
        CUDA_SOURCE_DIR = os.environ["CUDA_SOURCE_DIR"]

        cmake_args += [f"-DFMT_SOURCE_DIR={FMT_SOURCE_DIR}", f"-DGOOGLETEST_SOURCE_DIR={GOOGLETEST_SOURCE_DIR}", f"-DCUDA_SOURCE_DIR={CUDA_SOURCE_DIR}"]

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(["cmake", "--build", "."], cwd=self.build_temp, env=env)


with open("../readme.md", "r") as f:
    long_description = f.read()

# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="dmv_profiler",
    version="0.0.1",
    author="Suraj Kesavan",
    author_email="jarus3001@gmail.com",
    description="Data movement profiler that works for CUDA-profiling",
    long_description=long_description,
    packages=find_packages("."),
    package_dir={"": "."},
    ext_modules=[CMakeExtension("dmv_profiler")],
    python_requires=">=3.6",
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False
)