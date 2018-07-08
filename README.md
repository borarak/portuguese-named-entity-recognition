# portuguese-ner

A NER problem. The documents are in Portuguese and furthur documents are specific in nature: they are legal contracts. We are not interested in all named entities but only certain types that appear in the documents like Company names, CPFs, CNPJs, Shares of each person and the value of these shares.

# Installation

All Python dependencies are installable via the provided `requirements.txt` file. To install, simply do

`pip install requirements.txt`

The project has the following dependencies. 

1. [MITIE](https://github.com/mit-nlp/MITIE):  MITIE is a NLP framework from MIT for facilitating quick integration with 3rd party applications, hence the choice of the MITIE backend for the project. To make predictions only the python package is required (provided via `requirements.txt`)

    MITIE needs a corpora vector. I have already created the corpora vector and provided it as a deliverable. However in case you need to re-create the corpora vector please foolw the steps [here](https://github.com/mit-nlp/MITIE)
    
2. [Spacy](https://spacy.io/) 

    We use the provided Portuguese packages by Spacy for POS tagging
    
    `python -m spacy download pt`
    
    `python -m spacy download pt_core_news_sm`
    
    
3. [NLTK](http://www.nltk.org/): We use NLTK along with it's poruguese packages for Sentence and word tokenisations

    `python -m nltk.downloader all`

# Configuration

All project configurations are saved in the `config.py` file. You must set the parameters in this file according to your directory structure before proceeding.


# Preparing annotations

Once you annotate documents with [brat](http://brat.nlplab.org/) you would have to furthur post-process them before training. To post-process simply run

`python create_annotations_from_brat.py`


# Training

To train a NER run the trainer with the `-t` flag set

`python parse_clean_data.py -t True`

Training will take time depending on number of annotations in we want to identify and the number of documents to train on. So be patient. In general, training on ~2000 annotated sentences with 7 annotations to be classified took about 6 hours on a 36 core 70GB machine.


# Evaluation

To evaluate the NER run with the `-e` flag set

`python parse_clean_data.py -e True`

Evaluations will saved to the file set in `config.TEST_RESULTS_FILE`


# Confidence scores

You can manage the confidence score for each annotation type to set thresholds so as to reduce the number of false positives by setting the `SCORE` variables in `config.py`



