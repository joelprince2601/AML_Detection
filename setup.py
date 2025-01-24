from setuptools import setup, find_packages

setup(
    name="aml_detection",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pandas>=2.1.4',
        'numpy>=1.26.2',
        'scikit-learn>=1.3.2',
        'networkx>=3.2.1',
        'streamlit>=1.29.0',
        'openpyxl>=3.1.2',
        'protobuf==3.20.0',
        'plotly>=5.18.0',
    ],
) 