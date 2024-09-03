import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nn_fac",
    version="0.3.3",
    author="Marmoret Axel",
    author_email="axel.marmoret@irisa.fr",
    description="Nonnegative factorization toolbox.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.imt-atlantique.fr/a23marmo/nonnegative-factorization/",    
    packages=setuptools.find_packages(),
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.7"
    ],
    license='BSD',
    install_requires=[
        'nimfa',
        'numpy >= 1.18.0, <1.24', # Starting from version 1.24, the NTD algorithm (and potentially the others) does not work anymore. Realted to issue #1 https://github.com/ax-le/nn-fac/issues/1. Should be fixed in the future.
        'scipy >= 0.13.0',
        'tensorly == 0.6.0',
    ],
    python_requires='>=3.7',
)
