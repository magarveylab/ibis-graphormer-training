# IBIS BGC Training Scripts

## Dataset preparation
1. **Download Dataset**
    - The dataset is available on Zenodo under `ibis_bgc_datasets/siamese_classification_training`.
    - Extract and place the contents in `training/dat/siamese_classification_training`, maintaining the original directory structure.
2. **Regenerate Subgraphs (for custom datasets)**
    - Use the `create_graph_from_ibis_dir` function from `preprocessing/ibis_bgc/generate_graphs.py` to regenerate subgraphs from local IBIS files.

## Training
1. **Configure Training**
    - Modify the default arguments in train.py as needed. The training process supports multiple GPUs, which can be specified using CUDA_VISIBLE_DEVICES:
```
cd SiameseClassificationTraining
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -logger_entity new_user
```
2. **Checkpoint Management**
- Convert DeepSpeed checkpoints to PyTorch format for seamless loading in subsequent training steps:
```
python save.py
```
3. **Model Export**
- Export the trained model in TorchScript format for efficient and scalable inference. Before exporting, ensure the selected model parameters match those in the export script, updating them if necessary:
```
python export.py
```