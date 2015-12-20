#
# Experiment script for comparing static and adaptive effort metrics.
#
# [Reference]
# Jiepu Jiang and James Allan. Adaptive effort for search evaluation metrics.
# In Proceedings of the 38th European Conference on Information Retrieval (ECIR '16), 2016
#
# http://people.cs.umass.edu/~jpjiang/papers/ecir16_metrics.pdf
#

import numpy as np

from utils import *
from dataset import *
from query_metrics import *
from session_metrics import *

# load the dataset
session_ratings = load_ratings('data/session')
session_results = load_results('data/results')
session_qrels = load_qrels('data/qrels')

# three different effort vectors
evec_static = [1.0, 1.0, 1.0]
evec_param = [1.0 / 4, 1.0, 1.0]
evec_time = [9.8 / 37.6, 23.0 / 37.6, 1.0]

# the gs distribution used by GP, GAP, and GRBP
gs = [0, 0.4, 0.6]

# In the regression experiment, we generate $numsamples random partitions of the dataset,
# and perform x-fold cross validation on each partition, where x = $numfolds.
numfolds, numsamples = 10, 10

# k is the number of top ranked results for each query to be considered for evaluation.
# k = 9 because the dataset only provides 9 results per SERP.
k = 9

# the user metric to be compared with; umetric can be either 'performance' or 'difficulty' in this dataset.
umetric = 'performance'

# the best adaptive effort metric; baselines will be compared with this metric
best = SQMetric(GRBP(evec_param, 0.6, gs), np.mean)

# stores a list of metrics
metrics = []

metrics.append(
        [  # P@k
            'P',
            [
                SQMetric(Prec(evec_static), np.mean),
                SQMetric(Prec(evec_param), np.mean),
                SQMetric(Prec(evec_time), np.mean)
            ]
        ]
)

metrics.append(
        [  # average precision
            'AP',
            [
                SQMetric(AvgPrec(evec_static), np.mean),
                SQMetric(AvgPrec(evec_param), np.mean),
                SQMetric(AvgPrec(evec_time), np.mean)
            ]
        ]
)

metrics.append(
        [  # reciprocal rank
            'RR',
            [
                SQMetric(RR(evec_static), np.mean),
                SQMetric(RR(evec_param), np.mean),
                SQMetric(RR(evec_time), np.mean)
            ]
        ]
)

metrics.append(
        [  # graded P@k
            'GP',
            [
                SQMetric(GradPrec(evec_static, gs), np.mean),
                SQMetric(GradPrec(evec_param, gs), np.mean),
                SQMetric(GradPrec(evec_time, gs), np.mean)
            ]
        ]
)

metrics.append(
        [  # graded average precision
            'GAP',
            [
                SQMetric(GradAvgPrec(evec_static, gs), np.mean),
                SQMetric(GradAvgPrec(evec_param, gs), np.mean),
                SQMetric(GradAvgPrec(evec_time, gs), np.mean)
            ]
        ]
)

metrics.append(
        [  # rank-biased precision with p = 0.8
            'RBP (p=0.8)',
            [
                SQMetric(RBP(evec_static, 0.8), np.mean),
                SQMetric(RBP(evec_param, 0.8), np.mean),
                SQMetric(RBP(evec_time, 0.8), np.mean)
            ]
        ]
)

metrics.append(
        [  # rank-biased precision with p = 0.6
            'RBP (p=0.6)',
            [
                SQMetric(RBP(evec_static, 0.6), np.mean),
                SQMetric(RBP(evec_param, 0.6), np.mean),
                SQMetric(RBP(evec_time, 0.6), np.mean)
            ]
        ]
)

metrics.append(
        [  # graded rank-biased precision with p = 0.8
            'GRBP (p=0.8)',
            [
                SQMetric(GRBP(evec_static, 0.8, gs), np.mean),
                SQMetric(GRBP(evec_param, 0.8, gs), np.mean),
                SQMetric(GRBP(evec_time, 0.8, gs), np.mean)
            ]
        ]
)

metrics.append(
        [  # graded rank-biased precision with p = 0.6
            'GRBP (p=0.6)',
            [
                SQMetric(GRBP(evec_static, 0.6, gs), np.mean),
                SQMetric(GRBP(evec_param, 0.6, gs), np.mean),
                SQMetric(GRBP(evec_time, 0.6, gs), np.mean)
            ]
        ]
)

metrics.append(
        [  # expected reciprocal rank
            'ERR',
            [
                SQMetric(ERR(evec_static, 2), np.mean),
                SQMetric(ERR(evec_param, 2), np.mean),
                SQMetric(ERR(evec_time, 2), np.mean)
            ]
        ]
)

metrics.append(
        [  # DCG
            'DCG',
            [
                SQMetric(DCG(evec_static), np.mean),
                SQMetric(DCG(evec_param), np.mean),
                SQMetric(DCG(evec_time), np.mean)
            ]
        ]
)

metrics.append(
        [  # nDCG
            'nDCG',
            [
                SQMetric(NDCG(evec_static), np.mean),
                SQMetric(NDCG(evec_param), np.mean),
                SQMetric(NDCG(evec_time), np.mean)
            ]
        ]
)

metrics.append(
        [  # a variant of time-biased gain
            'TBG',
            [
                SQMetric(TBG([9.8, 23.0, 37.6], [0.26, 0.50, 0.55], [0, 0.2, 0.8], 31), np.mean)
            ]
        ]
)

metrics.append(
        [  # a variant of u-measure
            'U-measure',
            [
                SQMetric(UMeasure(2, [9.8, 23.0, 37.6], 65), np.mean)
            ]
        ]
)

metrics.append(
        [  # session-based DCG
            'sDCG',
            [
                SDCG(2, 4, True)
            ]
        ]
)

metrics.append(
        [  # normalized sDCG
            'nsDCG',
            [
                NSDCG(2, 4, True)
            ]
        ]
)

metrics.append(
        [  # estimated session nDCG
            'esnDCG',
            [
                ESNDCG(0.8, 0.7, False)
            ]
        ]
)

print(
    '%-20s  %20s  %20s  %20s  %20s  %20s  %20s'
    %
    (
        'Metric',
        'Pearson: static',
        'Pearson: param',
        'Pearson: time',
        'NRMSE: static',
        'NRMSE: param',
        'NRMSE: time'
    )
)

for [name, mets] in metrics:
    if len(mets) == 1:

        r, pr, rho, prho = correlation(session_ratings, session_results, session_qrels, umetric, mets[0], k)
        nrmse = regress(
                session_ratings, session_results, session_qrels,
                umetric, mets[0], k, 4.0, numfolds, numsamples
        )
        nrmse_best = regress(
                session_ratings, session_results, session_qrels,
                umetric, best, k, 4.0, numfolds, numsamples
        )

        print(
            '%-20s  %16.3f %-3s  %16s %-3s  %16s %-3s  %16.3f %-3s (p=%.3f)'
            %
            (
                name, r, star(pr), '', '', '', '',
                np.mean(nrmse), star(stats.ttest_rel(nrmse, nrmse_best)[1]),
                stats.ttest_rel(nrmse, nrmse_best)[1]
            )
        )

    else:

        r1, pr1, rho1, prho1 = correlation(session_ratings, session_results, session_qrels, umetric, mets[0], k)
        nrmse1 = regress(
                session_ratings, session_results, session_qrels,
                umetric, mets[0], k, 4.0, numfolds, numsamples
        )

        r2, pr2, rho2, prho2 = correlation(session_ratings, session_results, session_qrels, umetric, mets[1], k)
        nrmse2 = regress(
                session_ratings, session_results, session_qrels,
                umetric, mets[1], k, 4.0, numfolds, numsamples
        )

        r3, pr3, rho3, prho3 = correlation(session_ratings, session_results, session_qrels, umetric, mets[2], k)
        nrmse3 = regress(
                session_ratings, session_results, session_qrels,
                umetric, mets[2], k, 4.0, numfolds, numsamples
        )

        print(
            '%-20s  %16.3f %-3s  %16.3f %-3s  %16.3f %-3s  %16.3f %-3s  %16.3f %-3s  %16.3f %-3s      %-3s'
            %
            (
                name, r1, star(pr1), r2, star(pr2), r3, star(pr3),
                np.mean(nrmse1), '',
                np.mean(nrmse2), star(stats.ttest_rel(nrmse1, nrmse2)[1]),
                np.mean(nrmse3), star(stats.ttest_rel(nrmse1, nrmse3)[1]),
                star(stats.ttest_rel(nrmse2, nrmse3)[1])
            )
        )
