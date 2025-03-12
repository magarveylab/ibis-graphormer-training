

### Preparing the Data for IBIS-SM (Base)###
Due to the size of this dataset and Zenodo file hosting limits, users looking to retrain this model with the original genome data (or an augmented dataset) will need to re-generate the dataset as follows:
1. Download Genomes from the [NCBI FTP portal ](https://ftp.ncbi.nlm.nih.gov/) or using one of the [NCBI recommended methods](https://www.ncbi.nlm.nih.gov/guide/howto/dwn-records/) for large-scale downloads of data.
   - To download the genome files used in the paper, please refer to the assembly accessions provided in the `basename` column of `ibis_sm_datasets/biosyn_windows/genomes.csv` hosted on Zenodo
2.  A dataset of csvs representing the extracted biosynthetic windows of these genomes, containing chemotype and boundary annotations defined by PRISM and AntiSMASH is contained in the Zenodo archive `ibis_sm_datasets/biosyn_windows/`. Should you wish to expand the dataset, it will be necessary to run PRISM and AntiSMASH independently on external genomes, and provide similarly formatted csvs. 
3.  Use `preprocessing/ibis_sm/generate_graphs.py` to generate windowed genome graphs from these csvs.
    - NOTE: If you have your own dataset of genomes and all the prerequisite annotations (IBIS, PRISM, and optionally, antiSMASH), please instead use the `omnicons.graph_converters.homogeneous.genome_graph.get_biosynthetic_windows_from_orfs()` directly. You do not need to generate an intermediary csv file. This graph format is directly compatible with the tensor generation funtions in the `BiosyntheticWindowDataGenerator`.
4.  `training/ibis_sm/data_generator.py` Convert your windowed genome graphs into tensors and store them on disk. This step also patches in the embeddings and so resulting files can be very large. Be aware that the data occupies approximately 443 GiB of disk space. We suggest storing these files on-machine or using similar low-latency storage options to optimize training time, if possible.
5.  Proceed with training!

### Preparing the Data for IBIS-SM (MIBiG Fine-Tuned)###

Note: You do not need to complete the same data extraction procedures as above to proceed with mibig fine-tuning, provided you are happy to use the checkpoint from the original model, which is provided under `/training/dat/mibig_training/pretrained_checkpoints/ibis_sm.pt`
1. Regenerate the MIBiG graph files from the provided .csvs using `preprocessing/ibis_sm/mibig_training/generate_graphs.py`
2. Run [IBIS](https://github.com/magarveylab/ibis-publication/tree/main) to generate embeddings for all protein sequences. The protein embedding module alone is sufficient. This data has been prepared for the datasets used in this work and is hosted at the zenodo link under `ibis_sm_datasets/mibig_training/ibis_on_mibig3.1`
3. Generate the final training tensors for all MIBiG entries using `preprocessing/ibis_sm/mibig_training/generate_tensors.py`. The data splits have already been prepared for you, but the code for reproducing these is included in the file as well. 

### Training 

Set up Weights & Biases (wandb) by following the instructions [here](https://docs.wandb.ai/quickstart/).


Download and extract `ibis_sm_datasets.zip` from Zenodo, then move the contents to `training/dat`, preserving the existing file structure.

Training Order:
1. BiosyntheticWindowsTraining: Identifying BGCs in real-genome data, many chemotypes.
2. MibigTraining: BGC identification with MIBiG chemotypes on a dataset of known BGCs. Primarily for comparison with other tools.

Modify the default arguments in train.py as needed. The training process supports multiple GPUs; specify the target GPUs using CUDA_VISIBLE_DEVICES.
```
cd BiosyntheticWindowsTraining
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



