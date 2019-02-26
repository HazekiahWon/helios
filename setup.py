from setuptools import find_packages, setup
setup(name='helios',
      version='0.1',
      packages=find_packages(),
      py_modules=['common_imports','data','exp','files','image',
                  'logger','metrics','pipeline','tools'])