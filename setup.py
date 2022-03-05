import setuptools
import src

setuptools.setup(
    version=src.__version__,
    packages=setuptools.find_packages(),
    entry_points={"console_scripts": ["chimpstackr=src.run:main"]},
)