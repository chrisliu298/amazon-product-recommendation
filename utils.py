from collections import defaultdict


def top_n(predictions, n=10):
    top_n_items = defaultdict(list)
    for user_id, item_id, _, predicted_rating, _ in predictions:
        top_n_items[user_id].append((item_id, predicted_rating))
    for user_id, user_ratings in top_n_items.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n_items[user_id] = user_ratings[:n]
    return top_n_items


def save_ranking(predictions, file_path):
    with open(file_path, "w") as p_file:
        for user_id, user_ratings in top_n(predictions, n=10).items():
            for idx, (item_id, _) in enumerate(user_ratings):
                p_file.write(user_id + " " + item_id + " " + str(idx + 1) + "\n")
