import os
import glob

import spectraflow.data_io.input_parsing as input_parsing
import spectraflow.processing.peptide_processing as peptide_processing


# training and testing data
def read_plabel_input_directory(input_directory, configuration, collision, instrument, sample_limit):
    # Create instance of peptide extraction with the specified configuration
    peptide_extraction_object = input_parsing.ExtractPeptideInput(configuration=configuration)
    peptide_extraction_object.sample_limit = sample_limit
    train_test_input = {}

    # Possible to scale up to multiple folders
    print("Reading input folder: %s" % input_directory)
    # Use glob to retrieve input file (plabel) names in the input directory
    input_file_names = glob.glob(os.path.join(input_directory, "*.plabel"))

    for input_file in input_file_names:
        # Featurise input using peptide_data with the specified collision and instrument
        featurised_peptide_input = peptide_extraction_object.featurise_input_training(input_file, collision, instrument)
        train_test_input = peptide_processing.combine_groups(train_test_input, featurised_peptide_input)

    return train_test_input
