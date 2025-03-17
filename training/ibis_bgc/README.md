# IBIS BGC Training Scripts

## Dataset preparation
1. **Regenerate Subgraphs (for custom datasets)**
    - Use the `create_graph_from_ibis_dir` function from `preprocessing/ibis_bgc/generate_graphs.py` to regenerate subgraphs from local IBIS files.

## Training
1. **Set Up Weights & Biases (wandb)**
    - Follow the [official quickstart guide](https://docs.wandb.ai/quickstart/) to configure Weights & Biases for experiment tracking.
2. **Prepare the Dataset**
    - Download and extract `ibis_bgc_datasets.zip` from Zenodo.
    - Move the extracted contents to training/dat, ensuring the original file structure is preserved.
3. **Training Worflow**
    - Modify the default arguments in train.py as needed. The training process supports multiple GPUs, which can be specified using CUDA_VISIBLE_DEVICES:
```
cd SiameseClassificationTraining
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -logger_entity new_user
```
4. **Checkpoint Management**
    - Convert DeepSpeed checkpoints to PyTorch format for seamless loading in subsequent training steps:
```
python save.py
```
5. **Model Export**
    - Export the trained model in TorchScript format for efficient and scalable inference. Before exporting, ensure the selected model parameters match those in the export script, updating them if necessary:
```
python export.py
```