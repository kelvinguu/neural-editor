FROM kelvinguu/pytorch:1.2
# FROM tensorflow/tensorflow:0.12.0
# FROM continuumio/anaconda

# Add the PostgreSQL PGP key to verify their Debian packages.
# It should be the same key as https://www.postgresql.org/media/keys/ACCC4CF8.asc
RUN apt-key adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys B97B0AFCAA1A47F044F244A07FCC7D46ACCC4CF8

# Add PostgreSQL's repository. It contains the most recent stable release of PostgreSQL, ``9.3``.
RUN echo "deb http://apt.postgresql.org/pub/repos/apt/ precise-pgdg main" > /etc/apt/sources.list.d/pgdg.list

# Install ``python-software-properties``, ``software-properties-common`` and PostgreSQL 9.3
# There are some warnings (in red) that show up during the build. You can hide
# them by prefixing each apt-get statement with DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python-software-properties software-properties-common postgresql-9.3 postgresql-client-9.3 postgresql-contrib-9.3

RUN apt-get update
RUN apt-get --yes --force-yes install libffi6 libffi-dev libssl-dev libpq-dev git

RUN pip install --upgrade pip
RUN pip install jupyter
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension  # add Jupyter notebook extension

RUN pip install fabric
RUN pip install pyOpenSSL==16.2.0
RUN pip install psycopg2==2.6.1
RUN pip install SQLAlchemy==1.1.0b3
RUN pip install cherrypy==8.1.2
RUN pip install bottle==0.12.10
RUN pip install boto==2.43.0

RUN pip install requests
RUN pip install nltk==3.2.3
RUN python -m nltk.downloader punkt  # download tokenizer data

RUN pip install keras==1.1.0
RUN pip install pyhocon line_profiler pytest tqdm faulthandler python-Levenshtein gitpython futures jsonpickle prettytable tensorboard_logger click

RUN apt-get update
RUN apt-get install -y vim less tmux nmap
COPY .tmux.conf /root

# vim bindings for Jupyter
# https://github.com/lambdalisue/jupyter-vim-binding
RUN mkdir -p $(jupyter --data-dir)/nbextensions
RUN git clone https://github.com/lambdalisue/jupyter-vim-binding $(jupyter --data-dir)/nbextensions/vim_binding
RUN jupyter nbextension enable vim_binding/vim_binding

# autoreload for Jupyter
RUN ipython profile create
RUN echo 'c.InteractiveShellApp.exec_lines = []' >> ~/.ipython/profile_default/ipython_config.py
RUN echo 'c.InteractiveShellApp.exec_lines.append("%load_ext autoreload")' >> ~/.ipython/profile_default/ipython_config.py
RUN echo 'c.InteractiveShellApp.exec_lines.append("%autoreload 2")' >> ~/.ipython/profile_default/ipython_config.py

# just installing so we can get tensorboard
RUN pip install tensorflow

RUN pip install annoy pympler