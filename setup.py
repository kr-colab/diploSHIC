from setuptools import setup
from numpy.distutils.core import Extension, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

shic_stats = Extension(
    "diploshic.shicstats",
    sources=["diploshic/shicstats.pyf", "diploshic/utils.c"],
)
setup(
    name="diploSHIC",
    version="1.0.7",
    description="diploSHIC",
    python_requires="<3.12,>=3.10",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kr-colab/diploSHIC",
    author="Andrew Kern",
    author_email="adkern@uoregon.edu",
    license="MIT",
    packages=["diploshic"],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "scikit-allel",
        "scikit-learn",
        "tensorflow==2.15.0",
        "keras",
    ],
    scripts=[
        "diploshic/diploSHIC",
        "diploshic/makeFeatureVecsForChrArmFromVcfDiploid.py",
        "diploshic/makeFeatureVecsForChrArmFromVcf_ogSHIC.py",
        "diploshic/makeFeatureVecsForSingleMsDiploid.py",
        "diploshic/makeFeatureVecsForSingleMs_ogSHIC.py",
        "diploshic/makeTrainingSets.py",
    ],
    zip_safe=False,
    extras_require={
        "dev": [],
    },
    setup_requires=[],
    ext_modules=[shic_stats],
)
