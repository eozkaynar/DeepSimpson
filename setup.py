from setuptools import setup, find_packages
import os

setup(
    name='deepsimpson',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
    'torch==2.7.0',
    'torchvision==0.22.0',
    'scikit-learn==1.6.1',
    'tqdm==4.67.1',
    'numpy==2.2.5',
    'pandas==2.2.3',
    'opencv-python==4.11.0.86',
    'vidaug==1.5',
    'scikit-image==0.25.2',
    'click==8.1.8',
    'matplotlib==3.10.1'
    ],
    entry_points={
        'console_scripts': [
            'deepsim=deepsimpson.segmentation:run',
        ],
    },
    author='eozkaynar',
    description='DeepSimpson: LV Segmentation and EF Estimation from EchoNet-Dynamic',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/eozkaynar/DeepSimpson',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
