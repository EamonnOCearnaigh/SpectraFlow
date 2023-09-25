from setuptools import setup, find_packages

setup(
    name='spectraflow',
    version='1.0.0',
    packages=find_packages(),
    scripts=[
        'spectraflow/predict_spectra.py',
        'spectraflow/train_model.py',
        'spectraflow/test_model.py'
    ],
    python_requires='>=3.10',
    install_requires=[
        'tensorflow>=2.13.0',
        'numpy>=1.24.3',
        'scipy>=1.11.1',
        'pandas>=2.0.3',
        'matplotlib>=3.7.2'
    ],
    package_data={
        'spectraflow': ['data/**/*', 'models/**/*']
    }
)
