# Issue

-----

# Requirements

```
pip install -r requirements.txt
```

------

# Structure

```
BLIP
├── README.md
├── examples
│   ├── Task(ex: Imagecaption)
│   │   ├── Datamodule.py
│   │   ├── Dataset(ex: NICE...)
│   │   │   ├── dataset.py
│   │   │   ├── util.py   (util for dataset)
│   │   │   ├── test
│   │   │   └── valid
│   │   └── utils
│   │       ├── eval.py
│   │       └── util.py   (util for datamodule )
│   ├── __init__.py
│   ├── datamodule.py     (base datamodule. Inherit it)
│   └── utils
│       └── util.py
├── nn
│   ├── __init__.py
│   ├── model_templates.py
├── infer.py
├── train.py

```
---------
# Arguments

### Arguments for training
|argument|available|
| ------ | ------- |
| max_epochs | - |
| batch_size | - |
| lr | - |
| warmup_ratio | - |
| num_workers | - |
| devices | - |
| default_root_dir | - |

### Arguments for task(WIP)


### Arguments for neural network(WIP)


-------------------

# Training


----------------

# Inference
