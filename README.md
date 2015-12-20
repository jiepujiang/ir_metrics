
This repository includes data and source codes for replicating experiments conducted in the following two articles:

**REFERENCE**

Jiepu Jiang and James Allan. Correlation between system and user metrics in a session.
In Proceedings of the first ACM SIGIR Conference on Human Information Interaction and Retrieval (CHIIR '16),
Chapel Hill, North Carolina, USA, 2016.

http://people.cs.umass.edu/~jpjiang/papers/chiir16_metrics.pdf

Jiepu Jiang and James Allan. Adaptive effort for search evaluation metrics.
In Proceedings of the 38th European Conference on Information Retrieval (ECIR '16), 2016

http://people.cs.umass.edu/~jpjiang/papers/ecir16_metrics.pdf

**DEPENDENCY**

python (2.7.6)
numpy (1.10.2)
scipy (0.16.1)

I **guess** other versions of python 2.7.*, numpy, and scipy are fine, but I've only tested on the specified version.

**TO REPLICATE THE EXPERIMENT**

In order to replicate the experiment, simply run

```
python exp_ecir16.py
```

and

```
python exp_chiir16.py
```

**TO REUSE THE DATASET**

You can make use of dataset.py to load and reuse the dataset.

```
from dataset import *

session_ratings = load_ratings('data/session')
session_results = load_results('data/results')
session_qrels = load_qrels('data/qrels')
```

**TO REUSE THE EVALUATION LIBRARY**

query_metrics.py and session_metrics.py include evaluation libraries for evaluating a query and a session
exp_chiir16.py and exp_ecir16.py include examples of using the libraries.

