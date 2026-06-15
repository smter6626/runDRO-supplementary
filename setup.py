import sys
import sysconfig

from Cython.Build import cythonize
from setuptools import Extension, setup

_cc = sysconfig.get_config_var("CC") or ""
if sys.platform == "win32" and "gcc" not in _cc.lower() and "clang" not in _cc.lower():
    extra_compile_args = ["/O2", "/std:c++17"]
else:
    extra_compile_args = ["-O3", "-std=c++17"]

ext_modules = cythonize(
    [
        Extension(
            "fastcircuits.nodes",
            ["fastcircuits/nodes.pyx"],
            language="c++",
            extra_compile_args=extra_compile_args,
        ),
        Extension(
            "fastcircuits.gcw._common",
            ["fastcircuits/gcw/_common.pyx"],
            language="c++",
            extra_compile_args=extra_compile_args,
        ),
        Extension(
            "fastcircuits.gcw._gcw",
            ["fastcircuits/gcw/_gcw.pyx"],
            language="c++",
            extra_compile_args=extra_compile_args,
        ),
        Extension(
            "fastcircuits.gcw._cw",
            ["fastcircuits/gcw/_cw.pyx"],
            language="c++",
            extra_compile_args=extra_compile_args,
        ),
        Extension(
            "fastcircuits.gcw._expectation",
            ["fastcircuits/gcw/_expectation.pyx"],
            language="c++",
            extra_compile_args=extra_compile_args,
        ),
        Extension(
            "fastcircuits.gcw._esd",
            ["fastcircuits/gcw/_esd.pyx"],
            language="c++",
            extra_compile_args=extra_compile_args,
        ),
    ],
    compiler_directives={"language_level": "3"},
)

setup(ext_modules=ext_modules)
