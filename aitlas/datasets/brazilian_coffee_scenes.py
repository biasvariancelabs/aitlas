import glob
import os
import shutil

from .multiclass_classification import MultiClassClassificationDataset


LABELS = [
    "coffee",
    "noncoffee",
]


class BrazilianCoffeeScenesDataset(MultiClassClassificationDataset):

    url = "http://www.patreo.dcc.ufmg.br/wp-content/uploads/2017/11/brazilian_coffee_dataset.zip"

    labels = LABELS
    name = "Brazilian Coffee Scenes dataset"


# Function to convert the dataset into internal aitlas format
def prepare(root):

    # remapping function
    def trans(x):
        i = x.index(".")
        return x[:i], f"{x[i + 1:].strip()}.jpg"

    # create class folders
    os.makedirs(os.path.join(root, "coffee"), exist_ok=True)
    os.makedirs(os.path.join(root, "noncoffee"), exist_ok=True)

    # iterate folds
    folds = glob.glob(f"{root}/*.txt")
    for fold in folds:
        fold_name = fold.split(".txt")[0]
        with open(fold) as f:
            lines = f.readlines()
            for label, img in map(lambda x: trans(x), lines):
                # print(os.path.join(root, fold_name, img))
                # print(os.path.join(root, label, img))
                shutil.move(
                    os.path.join(root, fold_name, img), os.path.join(root, label, img)
                )

        if fold_name != "/":
            shutil.rmtree(fold_name)
