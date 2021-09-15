# Eigenthemes
Source code for "Low-rank Subspaces for Unsupervised Entity Linking"

## Detailed instructions to run the code
1. Clone this repository using `git clone https://github.com/blind-anonymous/eigenthemes.git`
2. Download [Anaconda](https://www.anaconda.com/distribution/#download-section) (64-bit Python 3.7 version)
    * The Anaconda installer would provide the following prompt: *'Do you wish the installer to initialize Anaconda3 by running conda init? [yes|no]'*. Answering *'yes'* would make your life simpler, as your *'bashrc'/'bash_profile'* would be automatically updated with all the environment variables properly set.
    * If you choose to answer *'yes'* in the previous step, please run `source <path-to-your bashrc or bash_profile>` to set all the environment variables properly in your currently active terminal.
3. Setup the virtual environment named `el` to install all the required dependencies
	`conda env create -f el.yml`
4. Activate the installed environment
	`conda activate el`
5. Download the *resources* (`data` and `embeddings`) available via [google drive](https://drive.google.com/drive/folders/1iRxfWpE9AabIoO5gFHpqIrFhAyPQ6IRq?usp=sharing) (no sign-in required)
    1. Unzip the *data.zip* file in the empty `data` directory provided with the code repository
    2. Unzip the *deepwalk_wikidata.pickle.zip* file in the empty `embeddings` directory provided with the code repository
6. Download the *resources*  for Le and Titov (pretrained `models`) available via [google drive](https://drive.google.com/drive/folders/11S2otREtrcevK_eCoc4yo2N190nBouxc?usp=sharing) (no sign-in required)
    1. Unzip the *models.zip* file in the empty `models` directory provided with the code repository   
    **Important Note: If you want to train the model from scratch, you have to remove the current saved model (if existent) using `rm -rf models/*`. Retrain the models using `bash train_taumilnd.sh`, which will train five different models on the train set**
7. **Reproducing results presented in Table-2**
    * **NameMatch Baseline**: Run `python namematch.py`. This script will produce the results for the name-matching baseline as described in the paper for each of the four datasets considered in this study.
    * **<img src="https://render.githubusercontent.com/render/math?math=\tau">MIL-ND by Le and Titov**: Run `bash evaluate_taumilnd.sh`. This script will produce the results for the state of the art <img src="https://render.githubusercontent.com/render/math?math=\tau">MIL-ND for each of the four datasets considered in this study. It also outputs the *mean* and *standard deviation* of precision@1 and MRR over five independent runs of <img src="https://render.githubusercontent.com/render/math?math=\tau">MIL-ND on the terminal.
    * **Eigen (Proposed Technique)**: Run `python unsupervised_el.py`. This script will produce the results for Eigen for all the four considered datasets. The description of Eigenthemes (Eigen) can be found in the paper.
    * The overall micro *Precision@1* and *MRR* is present in the 12th and 13th column of the results files. Additional information can be self-inferred, thanks to the descriptive header present in each output file.   
    **Important Note: The results are stored in the empty directory `results` provided with the code repository. Precomputed results for the aforementioned techniques for all the datasets have already been updated in `results` directory of the code repository. Also, the results filenames are self-explanatory.**
