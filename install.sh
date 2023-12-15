#!/bin/bash

tar -xvf run/static_data/encoding.tsv.tar.gz
mv encoding.tsv run/static_data/

git submodule init
git submodule update

export CUDA 

#python3 -m venv env
#source env/bin/activate
pip install --upgrade pip

pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu118 || exit 1
pip install --upgrade setuptools

echo "=> Installing cpp-hash..."
cd submodules/cpp-hash || exit 1
python setup.py install || exit 1
cd ../..

# Install specific versions of packages for VLLM compatibility
pip install lingua-language-detector tiktoken transformers scikit-learn nltk pyinflect accelerate openai textattack pandas dacite dahuffman argostranslate

pip install starlette==0.27.0

echo "=> Installing vllm..."
cd submodules/vllm || exit 1
python setup.py install || exit 1
cd ../..

pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu118 || exit 1

echo "=> Installing watermark-benchmark..."
python setup.py install || exit 1

pip install --upgrade openai httpcore
