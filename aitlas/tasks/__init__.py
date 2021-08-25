from .evaluate import EvaluateTask
from .extract_features import ExtractFeaturesTask
from .predict import PredictSegmentationTask, PredictTask
from .prepare import PrepareTask
from .spacenet5.evaluate import SpaceNet5EvaluateTask
from .spacenet5.infer_speed import SpaceNet5InferSpeedTask
from .spacenet5.plot_graph_over_image import SpaceNet5PlotGraphOverImageTask
from .spacenet5.predict import SpaceNet5PredictTask
from .spacenet5.prepare_imagery import SpaceNet5PrepareImageryTask
from .spacenet5.prepare_speedmasks import SpaceNet5PrepareSpeedMasksTask
from .spacenet5.skeletonize import SpaceNet5SkeletonizeTask
from .spacenet5.split import SpaceNet5SplitTask
from .spacenet5.wkt_to_graph import SpaceNet5WktToGraphTask
from .split import RandomSplitTask, StratifiedSplitTask
from .train import TrainAndEvaluateTask, TrainTask
