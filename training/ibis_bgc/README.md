### Training ###
Modify the default arguments in train.py as needed. The training process supports multiple GPUs; specify the target GPUs using CUDA_VISIBLE_DEVICES.
```
cd SiameseClassificationTraining
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -logger_entity new_user
```
Convert DeepSpeed checkpoints to PyTorch checkpoints to enable seamless checkpoint loading in subsequent training steps.
```
python save.py
```

Export the model in torchscript format to support efficient and scalable inference. Before exporting, please ensure that the model parameters you have selected are the same as those reflected in the export script and update as necessary.
```
python export.py
```

### IBIS-BGC Dataset ###

Relevant datasets are hosted via Zenodo under `ibis_bgc_datasets/siamese_classification_training` the contents of these files should be placed in `training/dat/siamese_classification_training`, preserving the underlying file structure.

IBIS-BGC is trained based on subgraphs pulled from IBIS-KG, which is not currenlty hosted or shared in its entirety due to file size considerations (~2 TiB). The graphs necessary to re-train/validate the model are hosted on Zenodo as `ibis_bgc_datasets/siamese_classification_training/graphs`. 