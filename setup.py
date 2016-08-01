from setuptools import setup
from setuptools import find_packages


setup(name='Keras',
      version='1.0.6',
      description='Deep Learning for Python',
      author='Francois Chollet',
      author_email='francois.chollet@gmail.com',
      url='https://github.com/fchollet/keras',
      download_url='https://github.com/fchollet/keras/tarball/1.0.6',
      license='MIT',
      install_requires=['theano', 'pyyaml', 'six'],
      extras_require={
          'h5py': ['h5py'],
      },
      MOD_author='David Leon (Dawei Leng)',
      zip_safe=False,
      packages=find_packages())
