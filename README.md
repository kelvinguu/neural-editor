# Neural editor

Source code accompanying our paper, "Generating Sentences by Editing Prototypes" ([paper](https://arxiv.org/abs/1709.08878), [slides](http://kelvinguu.com/posts/generating-sentences-by-editing-prototypes/)).

**Authors:** Kelvin Guu\*, Tatsunori B. Hashimoto\*, Yonatan Oren, Percy Liang
(\* equal contribution)

NOTES:
- The instructions below are still a work in progress.
    - If you encounter any problems, please open a GitHub issue, or submit a
    pull request if you know the fix!
- This code requires data directories that we have not uploaded yet.
- This is research code meant to serve as a reference implementation. We
do not recommend heavily extending or modifying this codebase for other
purposes.

If you have questions, please email Kelvin at `guu.kelvin` at `gmail.com`.

## Related resources

- [Reproducible experiments on CodaLab](https://worksheets.codalab.org/worksheets/0xa915ba2f8b664ddf8537c83bde80cc8c/)
- [A detailed description of the training algorithm](http://kelvinguu.com/public/projects/neural_editor_training.pdf)

## Datasets

- Yelp: [download from CodaLab](https://worksheets.codalab.org/bundles/0x99d0557925b34dae851372841f206b8a/)
- One Billion Word Benchmark: [download from CodaLab](https://worksheets.codalab.org/bundles/0x017b7af92956458abc7f4169830a6537/)

Each line of each TSV file is a (prototype, revision) edit pair, separated by a tab.

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
    
    # Download word vectors
    wget http://nlp.stanford.edu/data/glove.6B.zip  # GloVe vectors
    unzip glove.6B.zip -d word_vectors

    # Download expanded set of word vectors
    cd word_vectors
    wget https://worksheets.codalab.org/rest/bundles/0xa57f59ab786a4df2b86344378c17613b/contents/blob/ -O glove.6B.300d_yelp.txt
    # TODO: do the same for glove.6B.300d_onebil.txt
    cd ..
    
    # Download datasets into data directory
    wget https://worksheets.codalab.org/rest/bundles/0x99d0557925b34dae851372841f206b8a/contents/blob/ -O yelp_dataset_large_split.tar.gz
    mkdir yelp_dataset_large_split
    tar xvf yelp_dataset_large_split.tar.gz -C yelp_dataset_large_split
    # TODO: do the same for one_billion_split

    # TODO: install NLTK

    # our code uses this variable to locate the data
    export TEXTMORPH_DATA=$DATA_DIR
    ```

## Quick Start

Before you begin, be sure to set the `TEXTMORPH_DATA` environment variable (see "Setup" above).

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
$ python textmorph/edit_model/main.py configs/edit_model/edit_baseline.txt
```
- `textmorph/edit_model/main.py` is the main script for training an edit model.
- It takes a config file as input: `configs/edit_model/edit_baseline.txt`
    - The config file is in [HOCON format](https://github.com/lightbend/config/blob/master/HOCON.md).
- The script will dump checkpoints into `$DATA_DIR/edit_runs`
    - inside `edit_runs`, each training run is assigned its own folder.
    - The folders are numbered 0, 1, 2, ...
- `main.py` will complain if the Git working tree is dirty (because it logs the
current commit as a record of the code's current state)
    - to override this, pass `--check_commit disable`

