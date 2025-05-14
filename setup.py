from setuptools import setup, find_packages

setup(
    name='tts-phoneme-model',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A lightweight phoneme-based Text-to-Speech model.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/tts-phoneme-model',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch>=1.7.0',
        'numpy',
        'scipy',
        'librosa',
        'matplotlib',
        'pandas',
        'tqdm',
        'soundfile',
        'scikit-learn',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)