import setuptools
import src
# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="chimpstackr",
    version=src.__version__,
    author="Noah Peeters",
    author_email="apenz1.peeters@gmail.com",
    # description="Focus Stacking software",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://github.com/noah-peeters/ChimpStackr",
    project_urls={
        "Bug Tracker": "https://github.com/noah-peeters/ChimpStackr/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    entry_points={"console_scripts": ["app=src.run:main"]},
)