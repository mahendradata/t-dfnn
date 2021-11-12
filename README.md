
# T-DFNN

This repository is a proof of concept of algorithms described in [T-DFNN: An Incremental Learning Algorithm for Intrusion Detection Systems](https://github.com/datanduth/t-dfnn) paper.

## Project Guidelines

### Installing Anaconda
Install [Anaconda](https://www.anaconda.com/) by following [this guidelines](https://docs.anaconda.com/anaconda/install/). 

### Create conda environment

Create a conda environment and install the necessary packages.

    (base)$ conda create --name t-dfnn python=3.9
    (base)$ conda activate t-dfnn
    (t-dfnn)$ conda install -c conda-forge mamba=0.17.0
    (t-dfnn)$ mamba install -c conda-forge pandas=1.3.4
    (t-dfnn)$ mamba install -c conda-forge numpy=1.21.4
    (t-dfnn)$ mamba install -c conda-forge scikit-learn=1.0.1
    (t-dfnn)$ mamba install -c conda-forge matplotlib=3.4.3
    (t-dfnn)$ mamba install -c conda-forge scikit-multiflow=0.5.3
    
### Installing Tensorflow

If you don't have GPU, install Tensorflow package.

    (t-dfnn)$ mamba install -c conda-forge tensorflow=2.6.0

If you do have GPU, you can install Tensorflow GPU package.

    (t-dfnn)$ mamba install -c conda-forge tensorflow-gpu=2.6.0
    
Be aware that to use Tensorflow GPU, the following NVIDIA® software must be installed on your system:

-   [NVIDIA® GPU drivers](https://www.nvidia.com/drivers) — CUDA® 11.2 requires 450.80.02 or higher.
-   [CUDA® Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) — TensorFlow supports CUDA® 11.2 (TensorFlow >= 2.5.0) 

### Prepare the dataset

Run these two programs to download and preprocess the CICIDS2017.

    (t-dfnn)$ bash download.sh
    (t-dfnn)$ python preprocessing.py -o data/preprocessing.log \
			    CICIDS2017-MachineLearning/ \
			    dataset/

### Running other programs

After the dataset is ready, you can run other programs in this repository. For examples:

Run `T-DFNN` program:

    (t-dfnn)$ python t-dfnn.py conf/t-dfnn.json exp1

Run `DFNN-all` program:

    (t-dfnn)$ python dfnn-all.py conf/dfnn-all.json exp2

Run `DFNN-batch` program:

    (t-dfnn)$ python dfnn-batch.py conf/dfnn-batch.json exp3

Run `HoeffdingTree` program:

    (t-dfnn)$ python HoeffdingTree.py conf/HoeffdingTree.json exp4
 

## Folder Structure 

 - `conf`: contains configuration files in json format.
 - `ils`:
	 - `dataset`: contains libraries to preprocess and manipulate the dataset.
	 - `model`: contains libraries to create the models.
	 - `util`: contains utility libraries.

These are optional folders. You need to create this folder on your own.
- `dataset`: contains the preprocessed dataset. 
- `data`: a folder to save the experimental data.
