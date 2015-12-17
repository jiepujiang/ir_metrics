import numpy as np
import scipy.stats as stats

from utils import *
from query_metrics import *
from session_metrics import *


def correlation(sratings, sresults, sqrels, umetric, smetric, k):
    ratings = []
    sevals = []
    for sessid in sresults.keys():
        ratings.append(sratings[sessid][umetric])
        sevals.append(smetric.evaluate(sqrels[sessid], sresults[sessid], k))
    pearson, p_pearson = stats.pearsonr(ratings, sevals)
    spearman, p_spearman = stats.spearmanr(ratings, sevals)
    return pearson, p_pearson, spearman, p_spearman


def regress(sratings, sresults, sqrels, umetric, smetric, k, norm, numfolds, numsamples, seed=0):
    nrmse = []
    random.seed(seed)
    for i in xrange(0, numsamples):
        sessionlist = [sessid for sessid in sresults.keys()]
        random.shuffle(sessionlist)
        for foldid in xrange(0, numfolds):
            train, test = [], []
            for ix in xrange(0, len(sessionlist)):
                if ix % numfolds == foldid:
                    test.append(sessionlist[ix])
                else:
                    train.append(sessionlist[ix])
            nrmse.append(regress_fold(sratings, sresults, sqrels, umetric, smetric, k, norm, train, test))
    return nrmse


def regress_fold(sratings, sresults, sqrels, umetric, smetric, k, norm, train, test):
    ratings = []
    sevals = []
    for sessid in train:
        ratings.append(sratings[sessid][umetric])
        sevals.append(smetric.evaluate(sqrels[sessid], sresults[sessid], k))
    slope, intercept, _, _, _ = stats.linregress(sevals, ratings)
    sum_se = 0
    for sessid in test:
        rating = sratings[sessid][umetric]
        seval = smetric.evaluate(sqrels[sessid], sresults[sessid], k)
        rating_pred = slope * seval + intercept
        sum_se += (rating - rating_pred) ** 2
    return (sum_se / len(test)) ** 0.5 / norm


def star(pval):
    if pval < 0.001:
        return '***'
    elif pval < 0.01:
        return '**'
    elif pval < 0.05:
        return '*'
    else:
        return ''


session_ratings = load_ratings('rawdata/session')
session_results = load_results('rawdata/results')
session_qrels = load_qrels('rawdata/qrels')

evec_static = [1.0, 1.0, 1.0]
evec_param = [1.0 / 4, 1.0, 1.0]
evec_time = [9.8 / 37.6, 23.0 / 37.6, 1.0]

gs = [0, 0.4, 0.6]
numfolds, numsamples = 10, 10

k = 9

umetric = 'performance'

metrics = []

metrics.append(
        [
            'P',
            [
                SQMetric(Prec(evec_static), np.mean),
                SQMetric(Prec(evec_param), np.mean),
                SQMetric(Prec(evec_time), np.mean)
            ]
        ]
)

metrics.append(
        [
            'AP',
            [
                SQMetric(AvgPrec(evec_static), np.mean),
                SQMetric(AvgPrec(evec_param), np.mean),
                SQMetric(AvgPrec(evec_time), np.mean)
            ]
        ]
)

metrics.append(
        [
            'RR',
            [
                SQMetric(RR(evec_static), np.mean),
                SQMetric(RR(evec_param), np.mean),
                SQMetric(RR(evec_time), np.mean)
            ]
        ]
)

metrics.append(
        [
            'GP',
            [
                SQMetric(GradPrec(evec_static, gs), np.mean),
                SQMetric(GradPrec(evec_param, gs), np.mean),
                SQMetric(GradPrec(evec_time, gs), np.mean)
            ]
        ]
)

metrics.append(
        [
            'RBP (p=0.8)',
            [
                SQMetric(RBP(evec_static, 0.8), np.mean),
                SQMetric(RBP(evec_param, 0.8), np.mean),
                SQMetric(RBP(evec_time, 0.8), np.mean)
            ]
        ]
)

metrics.append(
        [
            'RBP (p=0.6)',
            [
                SQMetric(RBP(evec_static, 0.6), np.mean),
                SQMetric(RBP(evec_param, 0.6), np.mean),
                SQMetric(RBP(evec_time, 0.6), np.mean)
            ]
        ]
)

metrics.append(
        [
            'GAP',
            [
                SQMetric(GradAvgPrec(evec_static, gs), np.mean),
                SQMetric(GradAvgPrec(evec_param, gs), np.mean),
                SQMetric(GradAvgPrec(evec_time, gs), np.mean)
            ]
        ]
)

metrics.append(
        [
            'GRBP (p=0.8)',
            [
                SQMetric(GRBP(evec_static, 0.8, gs), np.mean),
                SQMetric(GRBP(evec_param, 0.8, gs), np.mean),
                SQMetric(GRBP(evec_time, 0.8, gs), np.mean)
            ]
        ]
)

metrics.append(
        [
            'GRBP (p=0.6)',
            [
                SQMetric(GRBP(evec_static, 0.6, gs), np.mean),
                SQMetric(GRBP(evec_param, 0.6, gs), np.mean),
                SQMetric(GRBP(evec_time, 0.6, gs), np.mean)
            ]
        ]
)

metrics.append(
        [
            'ERR',
            [
                SQMetric(ERR(evec_static, 2), np.mean),
                SQMetric(ERR(evec_param, 2), np.mean),
                SQMetric(ERR(evec_time, 2), np.mean)
            ]
        ]
)

metrics.append(
        [
            'DCG',
            [
                SQMetric(DCG(evec_static), np.mean),
                SQMetric(DCG(evec_param), np.mean),
                SQMetric(DCG(evec_time), np.mean)
            ]
        ]
)

metrics.append(
        [
            'nDCG',
            [
                SQMetric(NDCG(evec_static), np.mean),
                SQMetric(NDCG(evec_param), np.mean),
                SQMetric(NDCG(evec_time), np.mean)
            ]
        ]
)

metrics.append(
        [
            'TBG',
            [
                SQMetric(TBG([9.8, 23.0, 37.6], [0.26, 0.50, 0.55], [0, 0.2, 0.8], 31), np.mean)
            ]
        ]
)

metrics.append(
        [
            'sDCG',
            [
                SDCG(2, 4, True)
            ]
        ]
)

metrics.append(
        [
            'nsDCG',
            [
                NSDCG(2, 4, True)
            ]
        ]
)

metrics.append(
        [
            'esnDCG',
            [
                ESNDCG(0.8, 0.7, False, N=1000)
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

        print(
            '%-20s  %16.3f %-3s  %16s %-3s  %16s %-3s  %16.3f %-3s  %16s %-3s  %16s %-3s'
            %
            (
                name, r, star(pr), '', '', '', '',
                np.mean(nrmse), '',
                '', '',
                '', ''
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
            '%-20s  %16.3f %-3s  %16.3f %-3s  %16.3f %-3s  %16.3f %-3s  %16.3f %-3s  %16.3f %-3s'
            %
            (
                name, r1, star(pr1), r2, star(pr2), r3, star(pr3),
                np.mean(nrmse1), '',
                np.mean(nrmse2), star(stats.ttest_rel(nrmse1, nrmse2)[1] / 2),
                np.mean(nrmse3), star(stats.ttest_rel(nrmse1, nrmse3)[1] / 2)
            )
        )
