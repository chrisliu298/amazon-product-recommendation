import json
import random
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader, accuracy
from surprise.prediction_algorithms import *
from tqdm import tqdm

from metrics import *
from utils import *

seed = 272
random.seed(seed)
np.random.seed(seed)
warnings.filterwarnings("ignore")


models = {
    # baselines
    "Normal-Predictor": NormalPredictor(),
    "Baseline": BaselineOnly(verbose=False),
    # matrix factorization
    "SVD": SVD(verbose=False),
    "SVD++": SVDpp(verbose=False),
    "NMF": NMF(verbose=False),
    # k-nearest neighbors
    "KNN-basic": KNNBasic(verbose=False),
    "KNN-baseline": KNNBaseline(verbose=False),
    "KNN-means": KNNWithMeans(verbose=False),
    "KNN-zscore": KNNWithZScore(verbose=False),
    # misc
    "SlopeOne": SlopeOne(),
    "Coclustering": CoClustering(verbose=False),
}


def prepare_data(filename):
    data = pd.read_json(f"{filename}.json", lines=True)
    data_reduced = data[["reviewerID", "asin", "unixReviewTime", "overall"]]
    del data
    train, test = train_test_split(data_reduced, test_size=0.2, random_state=seed)
    train, val = train_test_split(train, test_size=0.2, random_state=seed)
    train.to_csv("train.tsv", index=False, sep="\t")
    val.to_csv("val.tsv", index=False, sep="\t")
    test.to_csv("test.tsv", index=False, sep="\t")


def train(train_dataset, val_dataset, test_dataset, test=False):
    val_metrics, test_metrics = [], []
    for model_name in tqdm(models):
        model = models[model_name]
        model.fit(train_dataset)
        val_metrics.append(evaluate((model_name, model), val_dataset, "val"))
        if test:
            test_metrics.append(evaluate((model_name, model), test_dataset, "test"))

    val_df = pd.DataFrame(val_metrics, index=list(range(len(val_metrics))))
    val_df.to_csv(f"val_metrics.tsv", index=False, sep="\t")
    if test:
        test_df = pd.DataFrame(test_metrics, index=list(range(len(test_metrics))))
        test_df.to_csv(f"test_metrics.tsv", index=False, sep="\t")


def evaluate(model, dataset, stage="test"):
    predictions = model[1].test(dataset)
    save_ranking(predictions, f"{stage}_{model[0]}_ranking.txt")

    precision, recall = precision_at_recall_k(predictions, k=10, threshold=2.5)
    metrics = {"Model": model[0]}
    metrics["MAE"] = accuracy.mae(predictions, verbose=False)
    metrics["RMSE"] = accuracy.rmse(predictions, verbose=False)
    metrics["Precision"] = sum(precision.values()) / len(precision)
    metrics["Recall"] = sum(recall.values()) / len(recall)
    metrics["Conversion rate"] = conversion_rate(predictions, k=10)
    metrics["F1 score"] = f1_score(precision, recall)
    metrics["NDCG"] = ndcg(predictions, 10)
    return metrics


if __name__ == "__main__":
    filename = "reviews_Video_Games_5"
    prepare_data(filename)
    reader = Reader(line_format="user item timestamp rating", sep="\t", skip_lines=1)
    train_dataset = Dataset.load_from_file(
        "train.tsv", reader=reader
    ).build_full_trainset()
    val_dataset = (
        Dataset.load_from_file("val.tsv", reader=reader)
        .build_full_trainset()
        .build_testset()
    )
    test_dataset = (
        Dataset.load_from_file("test.tsv", reader=reader)
        .build_full_trainset()
        .build_testset()
    )
    train(train_dataset, val_dataset, test_dataset, test=True)
