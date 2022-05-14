from collections import defaultdict

from scipy import sparse
from sklearn.metrics import ndcg_score


def precision_at_recall_k(predictions, k=10, threshold=3.5):
    predicted_rating_and_rating = defaultdict(list)
    for user_id, _, rating, predicted_rating, _ in predictions:
        predicted_rating_and_rating[user_id].append((predicted_rating, rating))
    precisions = {}
    recalls = {}
    for user_id, user_ratings in predicted_rating_and_rating.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        num_relevant = sum((rating >= threshold) for (_, rating) in user_ratings)
        num_recommendation = sum(
            (predicted_rating >= threshold)
            for (predicted_rating, _) in user_ratings[:k]
        )
        num_relevant_recommendation = sum(
            ((rating >= threshold) and (predicted_rating >= threshold))
            for (predicted_rating, rating) in user_ratings[:k]
        )
        precisions[user_id] = (
            num_relevant_recommendation / num_recommendation
            if num_recommendation > 0
            else 0
        )
        recalls[user_id] = (
            num_relevant_recommendation / num_relevant if num_relevant > 0 else 0
        )
    return precisions, recalls


def f1_score(precisions, recalls):
    precision = sum(precisions.values()) / len(precisions)
    recall = sum(recalls.values()) / len(recalls)
    return 2 * precision * recall / (precision + recall)


def conversion_rate(predictions, k=10, threshold=0):
    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, t, true_r, est, b in predictions:
        user_est_true[uid].append((est, true_r))
    precisions = dict()
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in user_ratings[:k]
        )

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

    return sum([1 if precision > 0 else 0 for precision in precisions.values()]) / len(
        precisions.values()
    )


def ndcg(predictions, k_highest_scores=10):
    ratings = [p.r_ui for p in predictions]
    predicted_ratings = [p.est for p in predictions]
    predicted_ratings = sparse.coo_matrix(predicted_ratings)
    ratings = sparse.coo_matrix(ratings)
    return ndcg_score(
        y_true=ratings.toarray(),
        y_score=predicted_ratings.toarray(),
        k=k_highest_scores,
    )
