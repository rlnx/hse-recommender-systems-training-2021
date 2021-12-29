import pandas as pd

def _metrics(predicted, ground_truth, k):
    if not ground_truth:
        return 0.0, 0.0

    predicted_k = predicted[:k]

    # Recall@k
    intersection = ground_truth.intersection(predicted_k)
    recall = len(intersection) / min(len(predicted_k), len(ground_truth))

    # Average Precision (AP@k)
    num_hits = 0.0
    ap_score = 0.0
    for i, pr in enumerate(predicted_k):
        if pr in ground_truth and pr not in predicted[:i]:
            num_hits += 1
            ap_score += num_hits / (i + 1.0)
    ap_score /= min(len(ground_truth), len(predicted_k))

    return recall, ap_score

def compute(predicted: pd.DataFrame, test, k=10):
    test_items_grouped = test[test.user_id.isin(predicted.user_id)] \
        .groupby('user_id') \
        .agg({'item_id': set})

    items_to_compare = test_items_grouped.merge(
        predicted.rename(columns={'item_id': 'predicted'}),
        on='user_id',
        how='left'
    )

    metrics = items_to_compare.apply(
        lambda row: _metrics(row.predicted, row.item_id, k),
        axis=1,
        result_type='expand'
    )
    metrics.rename(columns={ 0: 'recall', 1: 'map' }, inplace=True)
    return metrics.mean().to_dict()
