# Word2Vec Negative Sampling implementation

## About 
A project for training word embedding from wikipedia dataset using Word2Vec model, its negative sampling variation

## How to run
 - Install the conda environemnt using environment.yml file
 - Download text corpus (ending with .txt) to data folder
 - Change the DATA_PATH constant in main.py into the name of the root folder (containing all of your training data)
 - Run main.py (alternatively, change the epochs and learning_rate to achieve desirable result)

## Where to find the data and How to add it to folder
 - Download the enwiki-preprocessed.zip from [Data link](https://drive.google.com/file/d/1a5YjeMvGXgVpLAoIu6mv0DYo5snd-kQx/view?usp=sharing)
 - Unzip the folder and place it in "data" folder
 - If you follow these steps, you don't need to change anything in the "main.py" file