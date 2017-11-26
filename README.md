# Neural editor

Source code accompanying our paper [Generating Sentences by Editing Prototypes](https://arxiv.org/abs/1709.08878).

**Authors:** Kelvin Guu\*, Tatsunori B. Hashimoto\*, Yonatan Oren, Percy Liang
(\* equal contribution)

NOTES:
- This code requires data directories that we have not uploaded yet.
- This is research code meant to serve as a reference implementation. We
do not recommend heavily extending or modifying this codebase for other
purposes.

If you have questions, please email Kelvin at `guu.kelvin` at `gmail.com`.

## Setup

1. Install [Docker](https://www.docker.com/). If you want to use GPUs, also
install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

2. Download the repository and necessary data.
    ```bash
    DATA_DIR=$HOME/neural-editor-data
    REPO_DIR=$HOME/neural-editor

    # Download repository
    git clone https://github.com/kelvinguu/neural-editor.git $REPO_DIR

    # Set up data directory
    mkdir -p $DATA_DIR
    cd $DATA_DIR
    wget http://nlp.stanford.edu/data/glove.6B.zip  # GloVe vectors
    unzip glove.6B.zip -d word_vectors

    # TODO: download expanded set of word vectors
    # - glove.6B.300d_onebil.txt
    # - glove.6B.300d_yelp.txt

    # TODO: download datasets into data directory
    # - yelp_dataset_large_split
    # - onebillion_split

    # TODO: install NLTK

    # our code uses this variable to locate the data
    export TEXTMORPH_DATA=$DATA_DIR
    ```

## Quick Start

Start a Docker container:
```bash
$ python run_docker.py --root --gpu $CUDA_VISIBLE_DEVICES  # enter Docker
```
- `run_docker.py` pulls the latest version of our Docker image 
(kelvinguu/textmorph:1.2) and then starts an interactive Docker container.
    - `--root`: flag to run as root inside the container (optional)
    - `--gpu $CUDA_VISIBLE_DEVICES`: if you do not have a GPU, you can skip this
    argument.
- Inside the container, `$DATA_DIR` is [mounted](https://docs.docker.com/engine/admin/volumes/volumes/)
at `/data` and `$REPO_DIR` is mounted at `/code`.
- The current working directory will be `/code`.

Once you are inside the container, start a training run:
```bash
$ python textmorph/edit_model/main.py configs/edit_model/edit_onebil.txt
```
- `textmorph/edit_model/main.py` is the main script for training an edit model.
- It takes a config file as input: `configs/edit_model/edit_onebil.txt`
    - The config file is in [HOCON format](https://github.com/lightbend/config/blob/master/HOCON.md).
- The script will dump checkpoints into `$DATA_DIR/edit_runs`
    - inside `edit_runs`, each training run is assigned its own folder.
    - The folders are numbered 0, 1, 2, ...
- `main.py` will complain if the Git working tree is dirty (because it logs the
current commit as a record of the code's current state)
    - to override this, pass `--check_commit disable`

