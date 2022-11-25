# Packages Required
### 1. Install Python >= 3.7
### 2. Install the requirements, pip install -r requirements.txt

# Make sure your GPU supports cuda
### 1. Install CUDA toolkit
### 2. pip install --upgrade setuptools pip wheel
### 3. pip install nvidia-pyindex
### 4. pip install nvidia-cuda-runtime-cu11
### Check that pytorch is install with the correct CUDA version, with 11.6;
### pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
### pip install tf-nightly-gpu

# Running models
### Tests done on a single RTX3060, if you have access to multiple GPUs or industrial grade;
### Increase batch size to improve acc and loss
### Increase number of epochs
### Models are in NLPTests folder, run the python file to start training