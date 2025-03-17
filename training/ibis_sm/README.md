# IBIS SM Training Scripts
Training Graphormers to predict BGC boundaries

## Dataset preparation for Biosynthetic Window Training
Due to the large size of this dataset and the file hosting limitations on Zenodo, users intending to retrain the model using the original genome data or an augmented dataset must regenerate the dataset by following these steps:

1. **Download Genomic Data**
    - Obtain genome sequences from the [NCBI FTP portal](https://ftp.ncbi.nlm.nih.gov/) or use one of the [NCBI recommended methods](https://www.ncbi.nlm.nih.gov/guide/howto/dwn-records/) for large-scale data downloads.
    - To download the genome files used in the paper, please refer to the assembly accessions provided in the `basename` column of `ibis_sm_datasets/biosyn_windows/genomes.csv` hosted on Zenodo
2. **Obtain Biosynthetic Window Data**
    - A dataset of CSV files representing biosynthetic windows extracted from these genomes—annotated with chemotype and boundary information as defined by PRISM and antiSMASH—is available in the Zenodo archive under `ibis_sm_datasets/biosyn_windows/`.
    - To expand this dataset, users must run PRISM and antiSMASH independently on additional genomes and provide similarly formatted CSV files.
3. **Generate Windowed Genome Graphs**
    - Use  `preprocessing/ibis_sm/generate_graphs.py` to construct genome graphs from the extracted biosynthetic windows.
    - **Note**: If working with a custom genome dataset that includes all necessary annotations (IBIS, PRISM, and optionally antiSMASH), use the function `get_biosynthetic_windows_from_orfs` from `omnicons.graph_converters.homogeneous.genome_graph` directly. This eliminates the need to generate an intermediary CSV file, as the resulting graph format is fully compatible with tensor generation functions in `BiosyntheticWindowDataGenerator`.
4. **Convert Genome Graphs into Tensors**
    - Use `training/ibis_sm/data_generator.py` to convert the windowed genome graphs into tensor representations and store them on disk.
    - This step also integrates embeddings, significantly increasing file size. The processed data requires approximately 443 GiB of storage. To optimize training performance, it is recommended to store these files on a local machine or a low-latency storage solution.

## Dataset preparation for MiBIG Training
1. **Generate MiBIG Graph Files**
    - Use `preprocessing/ibis_sm/mibig_training/generate_graphs.py` to regenerate the MiBIG graph files from the provided .csv files.
2. **Generate Protein Embeddings with Ibis-Enzyme**
    - Run Ibis-Enzyme to generate embeddings for all protein sequences. Only the protein embedding module is required. The embeddings used in this work have already been precomputed and are available at the Zenodo repository under      `ibis_sm_datasets/mibig_training/ibis_on_mibig3.1`.
3. **Convert Genome Graphs into Tensors**
    - Use `preprocessing/ibis_sm/mibig_training/generate_tensors.py` to generate the final training tensors for all MiBIG entries. The dataset splits are pre-prepared, but the script also includes the necessary code for reproducing them if needed.

## Training

1. **Set Up Weights & Biases (wandb)**
    - Follow the [official quickstart guide](https://docs.wandb.ai/quickstart/) to configure Weights & Biases for experiment tracking.
2. **Prepare the Dataset**
    - Download and extract `ibis_sm_datasets.zip` from Zenodo.
    - Move the extracted contents to training/dat, ensuring the original file structure is preserved.
3. **Training Worflow**
    - **BiosyntheticWindowsTraining**: Identifies biosynthetic gene clusters (BGCs) in real genome data across multiple chemotypes.
    - **MibigTraining**: Trains on known BGCs from the MiBIG dataset, primarily for benchmarking against other tools.

Modify the default arguments in train.py as needed. The training process supports multiple GPUs, which can be specified using CUDA_VISIBLE_DEVICES:
```
cd BiosyntheticWindowsTraining
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -logger_entity new_user
```
4. **Checkpoint Management**
- Convert DeepSpeed checkpoints to standard PyTorch format to facilitate seamless loading in subsequent training steps:
```
python save.py
```
5. **Model Export**
- Export the trained model in TorchScript format for efficient and scalable inference. Before exporting, ensure the selected model parameters match those in the export script, updating them if necessary:
```
python export.py
```