import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gpr_complex",  # Replace with your own username
    version="0.1",
    author="Darlan Cavalcante Moreira",
    author_email="darcamo@gmail.com",
    description="A GPR library that can work with complex numbers",
    keywords="GPR complex",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/darcamo/gpr_complex",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers", "Topic :: Scientific/Engineering"
    ],
    python_requires='>=3.6',
    install_requires=["numpy", "scipy", "bokeh"])
