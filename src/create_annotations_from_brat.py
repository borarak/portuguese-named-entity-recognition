import os
import config


def get_annotations(file_name):

    """
    Gets all BRAT annotations from the data files
    :param file_name: The name of the data file containing text
    :return:
    """

    already_annotated = list()
    annotations_file_name = file_name.split(".")[0] + ".docx.ann"
    annotations_file = open(os.path.join(config.BRAT_ANNOTATIONS_DIR
                                         , annotations_file_name),"rb").readlines()
    data_file = open(os.path.join(config.BRAT_ANNOTATIONS_DIR,file_name)
                     ,"rb").read()

    # Clean existing err characters
    data_file = data_file.replace("<","").replace(">","")

    for line in annotations_file:
        tokens = line.split("\t")

        annotation_type = tokens[1].split(" ")[0].strip()
        annotated_entity = tokens[-1].strip()
        print "Current set: {}, current entity:{}".format(already_annotated, annotated_entity)
        if annotated_entity in already_annotated:
            print "Already annotated : {}".format(annotated_entity)
            continue
        else:
            already_annotated.append(annotated_entity)

        data_file = replace_annotations(annotated_entity, annotation_type, data_file)

    w_file = open(os.path.join(config.BRAT_POSTPROCESSED_DIR, file_name + "_brat"), "w")
    w_file.writelines(data_file)


def replace_annotations(annotated_entity, annotation_type, data_file):

    """
    Replaces all BRAT annotations in place with annotations tags in the data files
    :param annotated_entity: The entity to be annotated, e.g a name of a person
    :param annotation_type: The type of the entity, e.g "Person"
    :param data_file: the data file containing BRAT annotations (a copy)
    :return: in-placed annotated data file
    """

    replacement_str = " <{}> {} </{}> ".format(annotation_type, annotated_entity, annotation_type)
    print "Replacing {} with {}".format(annotated_entity, replacement_str)
    data_file = data_file.replace(annotated_entity, replacement_str)
    return data_file


def main():
    for file in os.listdir(config.BRAT_ANNOTATIONS_DIR):
        if file.endswith("txt"):
            print "Annotating {}".format(file)
            get_annotations(file)


if __name__ == "__main__":
    main()
