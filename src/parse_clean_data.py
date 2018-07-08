#!/usr/bin/env python
# -*- coding: utf-8 -*-


import nltk
from mitie import *
from nltk.tokenize import WordPunctTokenizer
import config
import re
import spacy
import argparse


def load_word_feature_extractor():

    """
    Loads an MITIE corpora vector. See README.md for details
    :return:
    """
    trainer = ner_trainer(config.CORPORA_VECTOR)
    return trainer


def get_sentence_tokenizer():

    """
    Returns an instance of a NLTK Potuguese sentence tokeniser
    :return:
    """
    sent_tokenizer=nltk.data.load(config.SENTENCE_TOKENIZER_POR)
    return sent_tokenizer


def doc_to_sentence(doc):

    """
    Tokenises a doc to sentences
    :param doc: A text document
    :return: an array of sentences
    """
    sent_tokenizer = get_sentence_tokenizer()
    return sent_tokenizer.sent_tokenizer.tokenize(doc)


def sentence_to_words(raw_text):

    """
    Tokenises a sentence to words
    :param raw_text:
    :return: an array of words
    """
    sent_tokenizer = get_sentence_tokenizer()
    return sent_tokenizer.sent_tokenizer.tokenize(raw_text)


def get_sentence_from_words(words):

    """
    Recreates a sentence from words
    :param words:
    :return:
    """
    return " ".join(words)


def get_default_entities(sentence):

    """
    We use all NNP POS tagged entities that are not explicitely labelled as "MISC"
    :param sentence: A text sentence
    :return: A dict of entities annotated with "MISC" tag along with their occurences
    """
    default_ann_tuples = list()
    spacy_pipeline = spacy.load('pt_core_news_sm')
    spacy_doc = spacy_pipeline(sentence)
    for word in spacy_doc:
        if len(word) > 0:
            if word.pos_ == "PROPN":
                entity_start_idx = word.idx
                entity_end_idx = word.idx + len(word)
                entity_type = word.ent_type_
                entity_word = word
                # print "{},{},{},{},{},{}".format(word.idx, word.idx + len(word), word, word.ent_type_, word.ent_iob_,
                #                                                    word.pos_)
                if entity_type == "MISC":
                    default_ann_tuples.append(("MISC", entity_start_idx, entity_end_idx))
    return default_ann_tuples


def extract_annotations_for_mitie(annotated_sentences, annotations):

    """
    Extracts annotations from training files
    :param annotated_sentences: Annotated source text sentences
    :param annotations: The annotations themselves e.g "Person", "CPF"
    :return: A dictionary of annotated entities with their occurences in the document
    """
    annotation_dict = {}
    for idx, sent in enumerate(annotated_sentences):
        sentence = sent
        ann_tuples = list()

        # Remove annotations while marking their indexes
        try:
            while sentence.index("<") > 0:
                start_idx = sentence.index("<")
                annotation = sentence[start_idx + 1]
                start_count = 0
                while start_count != 3:
                    start_count += 1
                    sentence.pop(start_idx)

                end_idx = sentence.index("</")
                end_count = 0
                while end_count != 3:
                    end_count += 1
                    sentence.pop(end_idx)

                # print "{},{},{}".format(annotation, start_idx, end_idx, sentence)
                if start_idx < end_idx:
                    if annotation in annotations:
                        ann_tuples.append((annotation, start_idx, end_idx, sentence))

        # When no more annotations found in doc, add to dict
        except ValueError:
            if len(ann_tuples) > 0:

                # Check if sentence has any other named entities, if so try to extract them
                # If the NNPs+ NER fall within out annotation, ignore
                # Otheriwse add an annotation for Misc
                default_ann_tuples = get_default_entities(get_sentence_from_words(sentence))

                # idx = sentence number
                annotation_dict[idx] = (ann_tuples, default_ann_tuples)
            continue
    print "Total annotation: {}".format(len(annotation_dict.keys()))
    return annotation_dict


def add_training_samples(annotation_dict, trainer):

    """
    Add all annotations collected to trainer
    :param annotation_dict: A dictionary containing all collected annotations for a sentence
    :param trainer: A NER trainer instance
    :return: trainer instance with added annotations
    """
    sample_count = 0
    for key in annotation_dict.keys():
        # print key
        annotation_tuples, default_annotation_tuples = annotation_dict[key]
        final_sent = annotation_tuples[0][3]
        # print "final sent: ", final_sent
        train_sample = ner_training_instance(final_sent)
        for tup in annotation_tuples:
            try:
                train_sample.add_entity(xrange(int(tup[1]), int(tup[2])), str(tup[0]))
                sample_count += 1
            except:
                continue

        # try to add default annotation tuples
        for tup in default_annotation_tuples:
            try:
                train_sample.add_entity((int(tup[1]), int(tup[2])), str(tup[0]))
            except:
                continue

        # Finally add the annotated sentence to the trainer
        trainer.add(train_sample)

    print "{} samples added to trainer".format(sample_count)
    return trainer


def tokenise_sentences(trainer):

    """
    Tokenises documents into sentences and words
    Adds all available annotations to the trainer instance
    @TODO: Refactor
    :param trainer:
    :return:
    """
    sent_tokenizer = get_sentence_tokenizer()
    annotations = config.ANNOTATIONS
    annotation_dict = {}

    for file in os.listdir(config.TRAIN_FILES_DIR):
        print "File name: {}".format(file)
        raw_text = open(os.path.join(config.TRAIN_FILES_DIR, file), 'rb').read().decode('UTF-8')
        sentences = sent_tokenizer.tokenize(raw_text)

        annotated_sentences = []

        # Find all sentences in doc with atleast one annotation,
        for sent in sentences:
            words = WordPunctTokenizer().tokenize(sent)

            # if sentence has any valid annotations, add to annotated sentences list
            if len(set(annotations).intersection(set(words))) > 0:
                annotated_sentences.append(words)

        print "Annotated sentences: {}".format(len(annotated_sentences))
        annotation_dict = extract_annotations_for_mitie(annotated_sentences, annotations)
        trainer = add_training_samples(annotation_dict, trainer)
    return annotation_dict


def train_and_save_ner(trainer):

    """
    Trains and saved a NER classifier to disk

    :param trainer: An instance of MITIE trainer
    :return: None
    """
    trainer.num_threads = config.TRAINING_CPU_THREADS
    print "training..."
    ner = trainer.train()
    ner.save_to_disk(config.MODEL_FILE)


def load_ner_model():

    """
    Loads a trained NER model from disk

    :return: An instance of trained NER model
    """
    return named_entity_extractor(config.MODEL_FILE)


def load_ner_model2():

    """
    Loads a trained NER model from disk

    :return: An instance of trained NER model
    """
    return named_entity_extractor(config.MODEL_FILE2)


def get_scores(tag, score):

    """
    Thresholds for removing low confidence predictions

    :param tag: The NER tag
    :param score: the confidence score of the NER tag
    :return:
    """
    if tag == "Person":
        if score >= config.PERSON_SCORE:
            return True
        else:
            return False
    elif tag == "CNPJ":
        if score >= config.CNPJ_SCORE:
            return True
        else:
            return False
    elif tag == "CPF":
        if score > config.CPF_SCORE:
            return True
        else:
            return False
    elif tag == "Company":
        if score >= config.COMPANY_SCORE:
            return True
        else:
            return False


def prepare_test_documents():

    """
    prepares the test documents for evaluation on trained NER model

    :return: A list of detected entities
    """
    ner = load_ner_model()
    ner2 = load_ner_model2()
    sent_tokenizer = get_sentence_tokenizer()

    detected_entities = list()
    for file in os.listdir(config.TEST_FILES_DIR):
        print "File name: {}".format(file)
        test_text = open(os.path.join(config.TEST_FILES_DIR, file), 'rb').read().decode('UTF-8')
        test_sentences = sent_tokenizer.tokenize(test_text)


        entity_list = list()
        complete_ner = {}
        for sent in test_sentences:

            words = WordPunctTokenizer().tokenize(sent)
            entities = ner.extract_entities(words)
            entities_shares_related = ner2.extract_entities(words)
            if len(entities) > 0:
                 for e in entities:
                    range = e[0]
                    tag = e[1]
                    score = e[2]
                    if tag == "MISC":
                        continue

                    if get_scores(tag, score):
                        entity_text = " ".join(words[i].encode('utf8') for i in range).upper()
                        entity_list.append((entity_text, tag)) # score
                        complete_ner[entity_text] = tag

            if len(entities_shares_related) > 0:
                for e in entities:
                    range = e[0]
                    tag = e[1]
                    score = e[2]
                    if tag == "Shares" or tag == "SharesValue":
                        if get_scores(tag, score):
                            entity_text = " ".join(words[i].encode('utf8') for i in range).upper()
                            entity_list.append((entity_text, tag))  # score
                            complete_ner[entity_text] = tag
        detected_entities.append((file, complete_ner))
    return detected_entities


def verify_sharesValue(shares_value):
    re.compile("[\d\\.]*")


def get_cmd_line_parser():

    """
    Parses the command line arguments
    :return:
    """
    parser = argparse.ArgumentParser(description='Parses command line parameters')
    parser.add_argument('-t', '--train', action="store", dest="train",
                      help="performs training", default=False)
    parser.add_argument('-e', '--evaluate', action="store", dest="evaluate",
                      help="performs evaluation", default=False)
    return parser.parse_args()


def main():
    args = get_cmd_line_parser()

    # If train flag is set, train
    if args.train:
        trainer = load_word_feature_extractor()
        annotation_dict = tokenise_sentences(trainer)
        trainer = add_training_samples(annotation_dict, trainer)
        train_and_save_ner(trainer)

    # if eval flag is set, get predictions
    elif args.evaluate:
        detected_entites = prepare_test_documents()
        w_file = open(config.TEST_RESULTS_FILE,"w")
        for e in detected_entites:
            ner_dict = e[1]
            for e in ner_dict.keys():
                entity_tag = ner_dict[e]
                entity_value = e
                w_file.write("Entity: {}  Type: {}  \n".format(entity_value, entity_tag))


if __name__ == "__main__":
    main()



