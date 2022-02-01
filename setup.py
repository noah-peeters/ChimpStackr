import setuptools, glob

setuptools.setup(
    name="chimpstackr",
    packages=setuptools.find_packages(),
)

# import glob, os, setuptools
# from pathlib import Path

# requirements = []
# if os.path.exists("requirements.txt"):
#     with open("requirements.txt") as f:
#         requirements = f.read().splitlines()
# else:
#     requirements = []

# if os.path.exists("LICENSE.txt"):
#     with open("LICENSE.txt") as f:
#         license_txt = f.read()
# else:
#     license_txt = ""

# with open("README.md", "r") as f:
#     readme = f.read()

# this = os.path.dirname(os.path.realpath(__file__))
# def read(name):
#     with open(os.path.join(this, name)) as f:
#         return f.read()

# setuptools.setup(
#     name="python-stacking-gui",
#     version="1.0",
#     maintainer="Peeters Noah",
#     maintainer_email="mail@example.com",  # "apenz1.peeters@gmail.com"
#     url="https://github.com/noah-peeters/PythonFocusStackingGui",
#     description="Free focus stacking software",
#     long_description=readme,
#     packages=setuptools.find_packages("src"),
#     package_dir={"": "src"},
#     scripts=glob.glob("src/*.py", recursive=True),
#     python_requires=">3.6",
#     license=license_txt,
#     install_requires=read("requirements.txt"),
# )
