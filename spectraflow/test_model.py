import os
import time

import spectraflow.rnn.rnn_model as rnn_model
import spectraflow.rnn.model_configurations as model_configurations
import spectraflow.processing.peptide_processing as peptide_processing
import spectraflow.data_io.load_input_train_test as load_input_train_test
import spectraflow.processing.testing_utils as testing_utils


def main():

    print("\n=-=-=-=-=-=-=-= Testing Beginning =-=-=-=-=-=-=-=\n")

    # Path management for portability
    # Get the directory containing script
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct paths to the data and model folders
    data_folder_path = os.path.join(script_directory, 'data')
    model_folder_path = os.path.join(script_directory, 'models')

    # Trained model path
    #model_folder = os.path.join(model_folder_path, "ICR_1")
    model_folder = os.path.join(model_folder_path, "ICR_2")
    model_path = os.path.join(model_folder, "baseline_model_weights.h5")

    # Testing data (plabel format) path(s), multiple folders accepted
    #testing_input_folder = os.path.join(data_folder_path, "input_testing", "ICR_1")
    testing_input_folder = os.path.join(data_folder_path, "input_testing", "ICR_2")
    # etc.

    # Path for output testing results file
    testing_output_folder = os.path.join(data_folder_path, "output_testing")
    testing_output_path = os.path.join(testing_output_folder, "ICR_2_testing.eps")

    # Create directories if needed.If directory already exists, does not raise exception.
    os.makedirs(testing_output_folder, exist_ok=True)

    # Get model configuration
    configuration = model_configurations.Configuration_HCD_All_Modifications()

    # get model instance
    model = rnn_model.SpecPred_Model(configuration)

    # build model
    print("Building model...")
    model_input_size = configuration.model_input_size
    model_output_size = configuration.get_tensorflow_output_size()
    number_layers = model.number_layers
    model.build_model(input_size=model_input_size, output_size=model_output_size, number_layers=number_layers)

    # load model weights
    print("Loading model weights...")
    model.load_model(model_path=model_path)

    # Testing input parameters
    collision = 0.35
    instrument = 'Lumos'
    sample_limit = 100000000

    # Print data regarding peptide mass groups
    print("Preparing peptides...")
    peptides = {}
    grouped_peptides = peptide_processing.combine_groups(peptides, load_input_train_test.read_plabel_input_directory(
        testing_input_folder, configuration, collision=collision, instrument=instrument, sample_limit=sample_limit))

    printer = peptide_processing.PeptideGroupsPrinter(grouped_peptides)
    printer.print_groups()

    # Start timer
    start_time = time.perf_counter()

    # Prediction process: Generate spectral data
    print("Generating spectra...")
    output_predicted_spectra = model.predict_spectra(grouped_peptides)

    # Compare spectra
    print("Comparing spectra...")
    pcc, cos, spc, kdt, SA = testing_utils.model_evaluation(output_predicted_spectra, grouped_peptides)

    # Model evaluation
    print("Evaluating model...")
    evaluation_metrics = ['PCC', 'COS', 'SPC', 'KDT', 'SA']
    thresholds = [0.6, 0.7, 0.8, 0.9]
    print(str(testing_utils.plot_metrics([pcc, cos, spc, kdt, SA], evaluation_metrics, thresholds, output_path=testing_output_path)))

    # Determine testing time
    test_time = time.perf_counter()

    print("Testing output location: ({})".format(testing_output_folder))
    print("Testing time: {:.3f}".format(test_time - start_time))
    print("\n=-=-=-=-=-=-=-= Testing complete =-=-=-=-=-=-=-=\n")


if __name__ == '__main__':
    main()