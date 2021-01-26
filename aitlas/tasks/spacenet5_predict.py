import os
import random

import numpy as np
import torch
from skimage import io

from aitlas.base import BaseTask
from aitlas.tasks.schemas import PredictTaskSchema


class SpaceNet5PredictTask(BaseTask):
    """Custom task for predicting with the SpaceNet5 model."""
    schema = PredictTaskSchema  # set up the schema for the task

    def __init__(self, model, config):
        """
        Parameters
        ----------
            model : BaseModel
                Which is to be evaluated.
            config : Config
                Specifying the details for this task.
        """
        super().__init__(model, config)

    def run(self):
        """Implements the main logic of this task."""

        def create_batched_indices(lst, batch_size):
            """
            [0, 1, 2, ... len(lst)] ->
                [[0, 1, ... batch_size],
                 [batch_size + 1, ... batch_size * 2],
                 [batch_size * 2 + 1, ... batch_size * 3],
                 ...
                 [batch_size * (num_batches - 1) + 1, ... len(lst)]
                ]
            """
            result = list()
            batch_ = list()
            for inx in lst:
                if len(batch_) == batch_size:
                    result.append(batch_)
                    batch_ = list()
                batch_.append(inx)
            if len(batch_) > 0:
                result.append(batch_)
            return result

        # Load dataset and model
        dataset = self.create_dataset(self.config.dataset_config)
        self.model.load_model(self.config.model_path)
        # Set model in evaluation mode so as to not change the weights
        self.model.eval()
        # Run predictions
        with torch.no_grad():
            n = dataset.__len__()
            indices = list(range(0, n)) # create indices
            if dataset.shuffle:
                random.shuffle(indices) # shuffle indices
            batches = create_batched_indices(indices, dataset.batch_size) # create batches of indices
            for batch in batches:
                # Transform a list of indices to a tensor of batched images
                input_images = list()
                for index in batch:
                    image, _ = dataset.__getitem__(index=index)
                    input_images.append(image)
                input_images = torch.tensor(np.array(input_images))
                # Run the prediction
                output_masks = self.model.predict(input_images)
                for (output_mask_index, output_mask) in enumerate(output_masks):
                    # Save the predicted mask
                    fp = os.path.join(self.config.output_path, dataset.get_filename(batch[output_mask_index]))
                    print (f"Saving at {fp}")
                    io.imsave(fp, (output_mask * 255).numpy().astype(np.uint8), compress=1)