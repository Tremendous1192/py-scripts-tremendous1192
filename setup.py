from setuptools import setup, find_packages

setup(
    name = 'py-scripts-tremendous1192',
    version = "0.0.16",
    description = "My scripts",
    long_description = "",
    author = 'Tremendous1192',
    license = 'MIT',
    classifiers=[
        "Development Status :: 1 - Planning"
    ],
    install_requires=[
        "matplotlib >= 3.7",
        "numpy >= 1.26",
        "pandas >= 2.2",
        'polars >=0.2',
        "seaborn >= 0.13",
        "scikit-learn >= 1.4"
    ]
)