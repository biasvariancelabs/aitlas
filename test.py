from aitlas.datasets import EurosatDataset
from aitlas.tasks import StratifiedSplitTask
from aitlas.models import ResNet50

# one tuple is (train, test)
splits = [(10, 90), (20, 80), (30, 70), (40, 60), (50, 50), (60, 40), (70, 30), (80, 20), (90, 10)]

dataset_path = "/home/dkocev/data/UCMerced"  # where is the dataset on disk
trainset_path = "/home/dkocev/data/UCMerced/train.csv"  # where to store the train IDs
testset_path = "/home/dkocev/data/UCMerced/test.csv"  # where to store the test IDs

# Loop through the splits, train and evaluate

results = []  # results accumulator

# iterate through the splits
for train, test in splits:
    # configure split task
    split_config = {
        "split": {
            "train": {
                "ratio": train,
                "file": trainset_path
            },
            "test": {
                "ratio": test,
                "file": testset_path
            }
        },
        "path": dataset_path
    }
    split_task = StratifiedSplitTask(None, split_config)
    split_task.run()

    # setup train set
    train_dataset_config = {
        "batch_size": 16,
        "shuffle": True,
        "num_workers": 4,
        "csv_file_path": trainset_path,
        "transforms": ["aitlas.transforms.ResizeCenterCropFlipHVToTensor"]
    }

    train_dataset = EurosatDataset(train_dataset_config)

    # setup test set
    test_dataset_config = {
        "batch_size": 16,
        "shuffle": False,
        "num_workers": 4,
        "csv_file_path": testset_path,
        "transforms": ["aitlas.transforms.ResizeCenterCropToTensor"]
    }

    test_dataset = EurosatDataset(test_dataset_config)
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

    # setup model
    epochs = 100
    model_directory = "./experiments/ucmerced/"
    model_config = {"num_classes": 21, "learning_rate": 0.001, "pretrained": False}
    model = ResNet50(model_config)
    model.prepare()

    # training and evaluation
    model.train_and_evaluate_model(
        train_dataset=train_dataset,
        epochs=epochs,
        model_directory=model_directory,
        val_dataset=test_dataset,
        run_id='2',
    )
    # collect results
    results.append(model.running_metrics.f1_score())
print(results)

# See the results

#df = pd.DataFrame(zip(splits, [round(float(r["Accuracy"]) * 100, 2) for r in results]),
#                  columns=["Train/Test", "F1 score"])

#print(df)



