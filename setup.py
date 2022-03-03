import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="chimpstackr",
    version="0.0.16",
    author="Noah Peeters",
    author_email="apenz1.peeters@gmail.com",
    description="Focus Stacking software",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/noah-peeters/ChimpStackr",
    project_urls={
        "Bug Tracker": "https://github.com/noah-peeters/ChimpStackr/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    entry_points={"console_scripts": ["run=src.run:main"]},
)