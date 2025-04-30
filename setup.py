from setuptools import setup, find_packages

setup(
    name='deepsimpson',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch==2.2.1',
        'torchvision==0.17.1',
        'scikit-learn==1.4.1.post1',
        'tqdm==4.66.2',
        'numpy==1.26.4',
        'pandas==2.2.1',
        'opencv-python==4.9.0.80',
        'vidaug==0.0.4',
        'scikit-image==0.22.0',
        'click==8.1.7',
        'matplotlib==3.8.4'
    ],
    entry_points={{
        'console_scripts': [
            'deepsim=deepsimpson.segmentation:run',
        ],
    }},
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
