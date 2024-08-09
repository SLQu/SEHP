# Scalable and Effective Negative Sample Generation for Hyperedge Prediction

# environment

conda create --name hedge python=3.9
conda activate hedge

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

pip install torch_scatter-2.1.0+pt113cu117-cp39-cp39-linux_x86_64.whl
pip install torch_sparse-0.6.16+pt113cu117-cp39-cp39-linux_x86_64.whl
pip install dgl-1.1.1+cu117-cp39-cp39-manylinux1_x86_64.whl
<!-- https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
torch_sparse-0.6.16+pt113cu117-cp310-cp310-linux_x86_64.whl -->
<!-- https://data.dgl.ai/wheels/cu113/repo.html
dgl-1.1.1+cu113-cp310-cp310-manylinux1_x86_64.whl -->

pip install scikit-learn
pip install torch_geometric
pip install torchmetrics
pip install gensim
pip install matplotlib
pip install ipdb
pip install pandas
pip install transformers
pip install accelerate -U
pip install sentencepiece
pip install tiktoken
pip install openai
pip install optuna
pip install configargparse
# Run code
source activate hedge
python run.py --batch_size 100 --info_stru 1 -- --negative_sample_mode dgns2
