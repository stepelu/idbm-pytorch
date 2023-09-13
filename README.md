# IDBM - PyTorch

This repository consists of a self-contained implementation (~500 lines of code, neural network model excluded) of the dataset transfer experiment of:

[_Diffusion Bridge Mixture Transports, Schrödinger Bridge Problems and Generative Modeling_](https://arxiv.org/abs/2304.00917).

The following assumptions are made (see the paper, specifically Section 5.4, for more details):

- the reference process is given by $dX_t = σdW_t$ over $t ∈ [0,1]$ for some scalar $σ ≥ 0$ ;
- the initial dataset is MNIST and the terminal dataset is a subset of EMNIST.

## Install

Having cloned this repository, the recommended installation procedure is as follows:

### 1. Create Virtual Environment

Create a new virtual environment and activate it.

For instance, using [(Mini)Conda](https://docs.conda.io/en/latest/miniconda.html):

```bash
conda create -n idbm pip
conda activate idbm
```

### 2. Install PyTorch

Install the latest appropriate version of PyTorch according to the [official instructions](https://pytorch.org/get-started/locally/).

### 3. Install Other Requirements

Install the remaining requirements:

```bash
pip install -r requirements.txt
```

## Run

The Python script [`idbm.py`](idbm/idbm.py) accepts the following options:

```bash
python idbm.py [FLAGS]

FLAGS:
    --method=METHOD
        Default: 'IDBM'
    --sigma=SIGMA
        Default: 1.0
    --iterations=ITERATIONS
        Default: 60
    --training_steps=TRAINING_STEPS
        Default: 5000
    --discretization_steps=DISCRETIZATION_STEPS
        Default: 30
    --batch_dim=BATCH_DIM
        Default: 128
    --learning_rate=LEARNING_RATE
        Default: 0.0001
    --grad_max_norm=GRAD_MAX_NORM
        Default: 1.0
    --ema_decay=EMA_DECAY
        Default: 0.999
    --cache_steps=CACHE_STEPS
        Default: 250
    --cache_batch_dim=CACHE_BATCH_DIM
        Default: 2560
    --test_steps=TEST_STEPS
        Default: 5000
    --test_batch_dim=TEST_BATCH_DIM
        Default: 500
    --loss_log_steps=LOSS_LOG_STEPS
        Default: 100
    --imge_log_steps=IMGE_LOG_STEPS
        Default: 1000
```

The findings of the paper are replicated by the following runs:

```bash
# IDBM -- Iterated Diffusion Bridge Mixture Transport:
python idbm.py --method=IDBM --sigma=1.0
python idbm.py --method=IDBM --sigma=0.5
python idbm.py --method=IDBM --sigma=0.2

# BDBM -- Backward Diffusion Bridge Mixture Transport:
python idbm.py --method=IDBM --sigma=1.0 --iterations=1 --training_steps=300000

# DIPF -- Diffusion Iterated Proportional Fitting Transport:
python idbm.py --method=DIPF --sigma=1.0
python idbm.py --method=DIPF --sigma=0.5
python idbm.py --method=DIPF --sigma=0.2
```

The runs' histories have been persisted on [Weights & Biases](https://wandb.ai/stepelu/pub-idbm-pytorch), to aid reproducibility, analysis and experimentation.
