import numpy as np
import scipy.stats as stats

from utils import *
from query_metrics import *
from session_metrics import *

session_ratings = load_ratings('data/session')

count_p = [0, 0, 0, 0, 0]

for rating in session_ratings.values():
    count_p[rating['performance'] - 1] += 1

print count_p

count_p = [0, 0, 0, 0, 0]

for rating in session_ratings.values():
    count_p[rating['difficulty'] - 1] += 1

print count_p