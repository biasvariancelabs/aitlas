## What is AiTLAS?

The AiTLAS toolbox (Artificial Intelligence Toolbox for Earth Observation) includes state-of-the-art machine learning methods for exploratory and predictive analysis of satellite imagery as well as repository of AI-ready Earth Observation (EO) datasets. It can be easily applied for a variety of Earth Observation tasks, such as land use and cover classification, crop type prediction, localization of specific objects (semantic segmentation), etc. 

The main goal of AiTLAS is to facilitate better usability and adoption of novel AI methods (and models) by EO experts, while offering easy access and standardized format of EO datasets to AI experts which allows benchmarking of various existing and novel AI methods tailored for EO data.

### Design

AiTLAS is designed such that leveraging recent (and sophisticated) deep-learning approaches over a variety of EO tasks (and data) is straightforward. On the one hand, it utilizes EO data resources in an AI-ready form; on the other hand, it provides a sufficient layer of abstraction for building and executing data analysis pipelines, thus facilitating better usability and accessibility of the underlying approaches - particularly useful for users with limited experience in machine learning, and in particular deep learning. 
    
It can be used both as an end-to-end standalone tool and as a modular library. Users can use and build on different toolbox components independently, be they related to the tasks, datasets, models, benchmarks, or complete pipelines. It is also flexible and versatile, facilitating the execution of a wide array of tasks on various domains and providing easy extension and adaptation to novel tasks and domains.

![AiTLAS Design](_media/aitlas_process.png)

AiTLAS is designed around the concept of a workflow, where users need to define a specific `task` be it an exploratory analysis of a dataset or a predictive task of a different kind, such as image classification, object detection, image segmentation, etc. The official [repository](https://github.com/biasvariancelabs/aitlas), includes many different [configuration files](https://github.com/biasvariancelabs/aitlas/tree/master/configs) for running various different workflows.

The instantiated task serves as an arbiter of the workflow and orchestrates the flow between the two central components of the toolbox - the datasets (`aitlas.datasets`) and the models (`aitlas.models`) - which relate to AI-ready formalized data and configurable model architectures, respectively. Programmatically, these modules are embedded within the core module `aitlas.base`, which contains all main abstract definitions related to every module, such as definitions of tasks, models, and datasets, but are also related to evaluations (`aitlas.metrics`), data transformations (`aitlas.transforms`), and various types of visualizations ( `aitlas.visulizataions` and `aitlas.datasets.visulizataions`). 
