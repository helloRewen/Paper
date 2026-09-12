from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from external_validation import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    SEED,
    build_sequence_model,
    build_tabular_model,
    choose_threshold,
)


def build_xgboost_model(y_train: np.ndarray) -> Pipeline:
    positive = max(1, int(np.sum(y_train == 1)))
    negative = max(1, int(np.sum(y_train == 0)))
    preprocessing = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC_COLS),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_COLS,
            ),
        ]
    )
    classifier = XGBClassifier(
        n_estimators=220,
        max_depth=3,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        min_child_weight=3,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=negative / positive,
        random_state=SEED,
        n_jobs=2,
    )
    return Pipeline([("prep", preprocessing), ("clf", classifier)])


def model_scores(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, dict[str, np.ndarray | float]]:
    y_train = train_df["anomaly_label"].to_numpy()
    results: dict[str, dict[str, np.ndarray | float]] = {}
    columns = NUMERIC_COLS + CATEGORICAL_COLS

    random_forest = build_tabular_model()
    random_forest.fit(train_df[columns], y_train)
    train_scores = random_forest.predict_proba(train_df[columns])[:, 1]
    results["tabular_rf"] = {
        "train_scores": train_scores,
        "test_scores": random_forest.predict_proba(test_df[columns])[:, 1],
        "threshold": choose_threshold(y_train, train_scores),
    }

    sequence = build_sequence_model()
    sequence.fit(train_df["path_text"], y_train)
    train_scores = sequence.predict_proba(train_df["path_text"])[:, 1]
    results["sequence_ngram_lr"] = {
        "train_scores": train_scores,
        "test_scores": sequence.predict_proba(test_df["path_text"])[:, 1],
        "threshold": choose_threshold(y_train, train_scores),
    }

    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform(train_df["transition_features"].tolist())
    x_test = vectorizer.transform(test_df["transition_features"].tolist())
    if x_train.shape[1] == 0 or len(np.unique(y_train)) < 2:
        train_scores = np.repeat(float(np.mean(y_train)), len(y_train))
        test_scores = np.repeat(float(np.mean(y_train)), len(test_df))
    else:
        graph = LogisticRegression(
            max_iter=1200,
            class_weight="balanced",
            solver="liblinear",
            random_state=SEED,
        )
        graph.fit(x_train, y_train)
        train_scores = graph.predict_proba(x_train)[:, 1]
        test_scores = graph.predict_proba(x_test)[:, 1]
    results["graph_transition_lr"] = {
        "train_scores": train_scores,
        "test_scores": test_scores,
        "threshold": choose_threshold(y_train, train_scores),
    }
    return results
