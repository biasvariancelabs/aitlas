from .evaluate import EvaluateTask
from .extract_features import ExtractFeaturesTask
from .predict import PredictSegmentationTask, PredictTask
from .prepare import PrepareTask
from .split import SplitTask
from .stats import StatsTask
from .train import TrainAndEvaluateTask, TrainTask
from .spacenet5_split import SpaceNet5SplitTask
from .spacenet5_02prepare_imagery import SpaceNet5PrepareImageryTask
from .spacenet5_03prepare_speedmasks import SpaceNet5PrepareSpeedMasksTask
from .spacenet5_04skeletonize import SpaceNet5SkeletonizeTask
from .spacenet5_05wkt_to_graph import SpaceNet5WktToGraphTask
from .spacenet5_06infer_speed import SpaceNet5InferSpeedTask
from .spacenet5_evaluate import SpaceNet5EvaluateTask
from .spacenet5_predict import SpaceNet5PredictTask