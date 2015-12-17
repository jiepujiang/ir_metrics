import numpy as np
import scipy.stats as stats

from utils import *
from query_metrics import *
from session_metrics import *

session_ratings = load_ratings('rawdata/session')
session_results = load_results('rawdata/results')
session_qrels = load_qrels('rawdata/qrels')

evec_static = [1.0, 1.0, 1.0]
umetric = 'performance'

metric = ERR(evec_static, 2)

for sessid in sorted(session_results.keys()):
    rating, qrels = session_ratings[sessid][umetric], session_qrels[sessid]
    qeval = []
    for qno in session_results[sessid].keys():
        results = session_results[sessid][qno]
        qeval.append(metric.evaluate(qrels, results, 9))
    print('%-10s%8.3f%8.3f' % (sessid, rating, np.mean(qeval)))

print('----------------------------------------------')

smetric = ESNDCG(1.0, 0.7, False, N=10000)

for sessid in sorted(session_results.keys()):
    rating, qrels = session_ratings[sessid][umetric], session_qrels[sessid]
    sresults = []
    for results in session_results[sessid].values():
        sresults.append(results)
    print('%-10s%8.3f%8.3f' % (sessid, rating, smetric.evaluate(qrels, sresults, 9)))
