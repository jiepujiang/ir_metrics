def load_results(path):
    results = dict()
    with open(path, 'r') as f:
        for line in f.readlines()[1:]:
            splits = line.rstrip('\n').split('\t')
            if len(splits) == 6:
                sessid, qno, _, url, _, _ = splits
                sessid, qno = int(sessid), int(qno)
                if sessid not in results:
                    results[sessid] = dict()
                if qno not in results[sessid]:
                    results[sessid][qno] = []
                results[sessid][qno].append(url)
            else:
                sessid, qno, _ = splits
                sessid, qno = int(sessid), int(qno)
                if sessid not in results:
                    results[sessid] = dict()
                if qno not in results[sessid]:
                    results[sessid][qno] = []
    f.close()
    for sessid in results.keys():
        results[sessid] = [results[sessid][qix + 1] for qix in xrange(0, len(results[sessid]))]
    return results


def load_qrels(path):
    qrels = dict()
    with open(path, 'r') as f:
        for line in f.readlines()[1:]:
            sessid, url, relevance = line.rstrip('\n').split('\t')
            sessid, relevance = int(sessid), int(relevance)
            if sessid not in qrels:
                qrels[sessid] = dict()
            qrels[sessid][url] = relevance
    f.close()
    return qrels


def load_ratings(path):
    ratings = dict()
    with open(path, 'r') as f:
        for line in f.readlines()[1:]:
            sessid, _, _, performance, difficulty = line.rstrip('\n').split('\t')
            sessid, performance, difficulty = int(sessid), int(performance), int(difficulty)
            if sessid not in ratings:
                ratings[sessid] = dict()
            ratings[sessid]['performance'] = performance
            ratings[sessid]['difficulty'] = difficulty
    f.close()
    return ratings
