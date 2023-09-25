import spectraflow.data_io.input_parsing as input_parsing


# Input format: peptide_sequence peptide_modifications charge
# + Normalised collision energy
# + Instrument used during mass spectrometry process
def read_input_file_prediction(input_file_prediction, configuration, collision, instrument):
    # Peptide data
    # Open input peptide file and read data
    print("Reading prediction input data")
    with open(input_file_prediction) as f:
        # Skip file header
        next(f)
        # Extract non-empty lines as input peptides
        input_peptides = [line.strip().split("\t") for line in f if line.strip()]

    # Extract peptide features
    peptide_extraction_object = input_parsing.ExtractPeptideInput(configuration=configuration)
    # Featurise
    prediction_input = peptide_extraction_object.featurise_input_prediction(input_peptides, collision, instrument)

    return prediction_input
