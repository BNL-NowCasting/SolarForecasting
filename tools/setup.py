from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

# sourcefiles = ['example_cy.pyx']
# extensions = [Extension("example_cy", sourcefiles)]
# setup(ext_modules = cythonize(extensions))

# sourcefiles = ['threshold.pyx']
# extensions = [Extension("threshold", sourcefiles, extra_compile_args=["--std=c99"])]
# setup(ext_modules = cythonize(extensions, language = "c++"))
# # setup(ext_modules = cythonize(extensions, language = "c++"), include_dirs=[numpy.get_include()])


sourcefiles = ['rolling.pyx']
extensions = [Extension('rolling', sourcefiles,include_dirs=[numpy.get_include()])]
setup(ext_modules = cythonize('rolling.pyx'))


