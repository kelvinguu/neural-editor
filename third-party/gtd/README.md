# PostgreSQL setup

Follow [these steps](https://www.codefellows.org/blog/three-battle-tested-ways-to-install-postgresql/)
to install Postgres on your local machine.

- (NOTE: if you skip this step, you won't be able to install the `psycopg2`
  Python package that our project requires)
- to make your life easier, check out the instructions for getting Postgres to
  launch automatically on startup

## Set up a local test database

(Only necessary if you're going to run DB-related unit tests)

```
$ createdb test_db
```

## Connecting to the central database on jonsson.stanford.edu

All you need to do is create an SSH tunnel.

```
$ ssh -N -n -L 15432:jonsson.stanford.edu:5432 kgu@jamie.stanford.edu  # (replace kgu with your own username)
```

This will make it seem as if the database at `jonsson.stanford.edu:5432` is
available at `localhost:15432`. All of our code assumes this.

# Python setup

If you're working on the NLP cluster, make sure you source
`/u/nlp/bin/setup.bash` as the last step in your `.bashrc`. Importantly, it adds
`/u/nlp/packages/pgsql/lib` to `$LD_LIBRARY_PATH`, which is needed by
`psycopg2`, Python's PostgreSQL library.

Create a virtualenv
```
$ virtualenv venv
```

(For the NLP cluster, you must actually use the default system Python's
virtualenv, NOT the new default set by `setup.bash`)
```
$ /usr/bin/virtualenv-2.7 venv

```

> NOTE: not every machine on the NLP cluster has virtualenv. You will
have to install it from one that does, such as jamie. Because all the
machines have roughly the same configuration, the venv will hopefully
work on all machines.

Activate it:
```
$ source venv/bin/activate
```

Upgrade pip:
```
$ pip install --upgrade pip
```

Install the right version of TensorFlow, depending on your OS

- For Mac OS X, CPU only, Python 2.7:
    - `export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.10.0rc0-py2-none-any.whl`
- For Ubuntu/Linux 64-bit, GPU enabled, Python 2.7 (Requires CUDA toolkit 7.5 and CuDNN v4.)
    - `export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0rc0-cp27-none-linux_x86_64.whl`

```
$ pip install --ignore-installed --upgrade $TF_BINARY_URL
```

Install additional packages
```
$ pip install -r requirements.txt
```

To exit the virtual environment:
```
$ deactivate
```

# CoreNLP setup
Clone the git repo
```
$ git clone git@github.com:stanfordnlp/CoreNLP.git
$ cd CoreNLP
```

Compile the jar file
```
$ ant  # compile (create class files)
$ cd classes  # create jar file
$ jar cf ../stanford-corenlp.jar edu  # compile jar
$ cd ..
```

Download model files:

- models jar from here: http://nlp.stanford.edu/software/stanford-english-corenlp-models-current.jar
- wikidict.tab.gz from here: http://nlp.stanford.edu/data/kbp/wikidict.tab.gz
  (warning, despite its name, it may not be gzipped and you may need to gzip it
  yourself)
- put the downloaded files in the root directory

Start the server
```
$ java -mx10g -cp "*:lib/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer  # launch server
```

Note: the first time you send a query to CoreNLP, it can take 3-5 minutes. This
is because the CoreNLP server must load 3-4 Gb worth of Wikidict for entity
linking.

# Running unit tests

In the root directory of your repository, run pytest:

```
$ py.test
```

To only run the tests in a particular module, pass its path, e.g.:

```
$ py.test semgen/tests/test_alignment.py
```

We make use of pytest fixtures. See [this link](http://docs.pytest.org/en/latest/fixture.html#fixtures) for more info.

# Install other command line tools:

Certain build and deployment tools inside `gtd.io` require:

- tmux
- autossh

# gtd.ml

This package defines a framework for implementing models in TensorFlow and
optionally, Keras.

The basic building block of the framework is the Model class. See its docstring
for more information.

# gtd.persist

This module contains objects for persisting data (to disk or a database). The
two main classes are:

- LazyMapping
- LazyIterator

The other classes primarily just support these. Each of these is an Abstract
Base Class (ABC), meant for you to subclass.

## LazyMapping

A LazyMapping implements collections.Mapping (i.e. it behaves like a dict).

```python
lazy_map = SomeLazyMappingSubclass(...)
val = lazy_map[key]
```

Instead of manually adding key-value pairs to a LazyMapping, you instead
provide a function which maps keys to values.

When you try to get a value from the LazyMapping, it first checks its cache to
see if the value has already been computed.

- if the value is present, LazyMapping returns the cached value
- if the value is not present, LazyMapping computes the value using your provided function, then stores it in the cache

To subclass LazyMapping, you just need to implement the abstractmethod
`compute_batch`. This should take a list of keys, and return a corresponding
list of values.

### Batch methods

For each method supported by a collections.Mapping, LazyMapping has a
corresponding batch version:

- `__getitem__`: `get_batch`
- `__contains__`: `contains_batch`

### Choosing a cache

To construct a LazyMapping you must provide a cache. LazyMapping expects the
cache to be a collections.MutableMapping (actually, it expects the cache to be a gtd.persist.BatchMapping, which extends collections.Mapping a little bit to
require two additional batch-related methods).

You almost never need to implement your own cache. The typical cache you want to use is TableMapping. TableMapping is a MutableMapping (behaves like a dict),
which stores its data in a PostgreSQL database.

To store objects in a database, TableMapping needs to know how to convert
objects into database rows. You specify this by defining an gtd.persist.ORM
("object relational mapping"). A TableMapping takes an ORM for keys and an ORM
for values.

### Pre-populating the cache

Sometimes, you just want to make sure that certain values are in the cache,
rather than actually retrieving them. In this case, rather than calling
`get_batch(keys)`, you can call `ensure_batch(keys)`. This is useful when
checking the presence of keys is much faster than retrieving the actual values.

### Handling failures

Sometimes your function for `compute_batch` will fail to compute the value for
certain keys in the batch. You can always just raise an Exception, terminating
computation.

However, if you'd still like to return a list of values so that you can
continue computation, you should return gtd.utils.Failure objects for the
values that you could not compute.

### Unit testing

For unit-testing, you can use the much simpler gtd.persist.SimpleBatchMapping
as your cache, which is actually just a very light wrapper over an actual
Python dict.

## LazyIterator

LazyIterator implements collections.Iterator (you can loop over it like a list,
and call next on it).

```python
it = SomeLazyIteratorSubclass(...)
val0 = next(it)
val1 = next(it)

for val in it:
    print val
```

It is lazy in the same spirit as LazyMapping. Rather than manually specifying a
sequence to iterate over, you provide  a function which describes how to
compute additional values.

When you call `next` on a LazyIterator, it first checks if the next value in
the sequence is present in its cache.

- if the value is present, LazyIterator just returns that value if not,
- LazyIterator uses your provided function to compute the next value, then
  appends it to the cache

To subclass LazyIterator, you just need to implement the abstractmethod
`compute_batch`. This should take a batch_size k, and return the next k values
from the iterator.

### Batch methods

For each method supported by a collections.Iterator, LazyIterator has a
corresponding batch version:

- `__next__`: `next_batch`

### Choosing a cache

To construct a LazyIterator, you must provide a cache. LazyIterator expects the
cache to be a gtd.persist.AppendableSequence.

You almost never need to implement your own cache. The typical choice is either
gtd.persist.FileSequence or gtd.persist.ShardedSequence.

FileSequence implements collections.Sequence (i.e. it behaves like a Python
list), which stores its data in a file on disk. ShardedSequence is just like
FileSequence, except it breaks its data into multiple files (shards).

To store data on disk, FileSequence needs to know how to serialize objects to
single-line strings.

You specify this by providing a gtd.persist.FileSerializer.

### Pre-populating the cache

Sometimes, you just want to make sure that certain values are in the cache,
rather than actually retrieving them. In this case, you can check the
LazyIterator's `num_computed` property to see how many values have been cached.

If not enough have been computed, you can advance the iterator immediately to
the last cached value, then keep calling `next` until the cache is
appropriately populated.

```python
it.advance_to(it.num_computed)
for k in range(...):
    next(it)
```

### Unit testing

For unit-testing, you can use the much simpler
gtd.persist.SimpleAppendableSequence as your cache, which is actually just a
very light wrapper over an actual Python list.
