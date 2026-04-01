def evaluate_shadow(production_log, shadow_log, criteria):
    import math

    n = len(production_log)

    # Alinear por input_id
    shadow_by_id = {r['input_id']: r for r in shadow_log}

    prod_correct = 0
    shad_correct = 0
    agreements   = 0
    shad_latencies = []

    for p in production_log:
        iid = p['input_id']
        s   = shadow_by_id[iid]

        if p['prediction'] == p['actual']:
            prod_correct += 1
        if s['prediction'] == s['actual']:
            shad_correct += 1
        if p['prediction'] == s['prediction']:
            agreements += 1

        shad_latencies.append(s['latency_ms'])

    prod_accuracy  = prod_correct / n
    shad_accuracy  = shad_correct / n
    accuracy_gain  = shad_accuracy - prod_accuracy
    agreement_rate = agreements / n

    # P95 nearest-rank
    shad_latencies.sort()
    p95_idx = math.ceil(0.95 * n) - 1
    shad_latency_p95 = shad_latencies[p95_idx]

    metrics = {
        'shadow_accuracy':     shad_accuracy,
        'production_accuracy': prod_accuracy,
        'accuracy_gain':       accuracy_gain,
        'shadow_latency_p95':  shad_latency_p95,
        'agreement_rate':      agreement_rate,
    }

    promote = (
        accuracy_gain  >= criteria['min_accuracy_gain'] and
        shad_latency_p95 <= criteria['max_latency_p95'] and
        agreement_rate >= criteria['min_agreement_rate']
    )

    return {'promote': promote, 'metrics': metrics}