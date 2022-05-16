import os
import sys
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import roc_auc_score, accuracy_score
from argparse import ArgumentParser

TEST_SIZE = 0.2
RAND_STATE = 42


def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--name", "-n", help="Experiment name (for saving checkpoints and submits).",
                        default="baseline")
    parser.add_argument("--data", "-d", help="Path train dataset.",
                        default='data/heart_cleveland_upload.csv')
    return parser.parse_args()


def main(args):
    os.makedirs(args.name, exist_ok=True)
    data = pd.read_csv(args.data)

    y = data['condition']
    x = data.drop(columns=['condition'], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=TEST_SIZE, random_state=RAND_STATE)

    model = LogisticRegression(solver='liblinear', random_state=RAND_STATE)
    model.fit(x_train, y_train)
    print(f"ROC-AUC score: {cross_val_score(model, x, y, scoring='roc_auc').mean()}")
    y_pred = model.predict(x_test)
    pd.DataFrame(y_pred).to_csv(f"{args.name}/submission.csv", index=False)


if __name__ == '__main__':
    args = parse_arguments()
    sys.exit(main(args))
