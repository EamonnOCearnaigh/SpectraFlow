import os
import time
import argparse

import spectraflow.rnn.rnn_model as rnn_model
import spectraflow.data_io.load_input_prediction as load_input
import spectraflow.data_io.prediction_output as prediction_output
import spectraflow.rnn.model_configurations as model_configurations


def main():

    print("\n=-=-=-=-=-=-=-= Prediction Beginning =-=-=-=-=-=-=-=\n")

    # Path management for portability
    # Get the directory containing script
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct paths to the data and model folders
    data_folder_path = os.path.join(script_directory, 'data')
    model_folder_path = os.path.join(script_directory, 'models')

    # Trained model path
    model_folder = os.path.join(model_folder_path, "ICR_1")
    #model_folder = os.path.join(model_folder_path, "ICR_2")
    model_path = os.path.join(model_folder, "baseline_model_weights.h5")

    # Input argument defaults/custom inputs
    # These values are used if command line arguments not passed

    # Path for input peptide file
    prediction_input_folder = os.path.join(data_folder_path, "input_prediction")
    prediction_input_path = os.path.join(prediction_input_folder, "example_peptide.txt")

    # Path for output prediction file
    prediction_output_folder = os.path.join(data_folder_path, "output_prediction")
    prediction_output_path = os.path.join(prediction_output_folder, "ICR_1_example_prediction.mgf")
    #prediction_output_path = os.path.join(prediction_output_folder, "ICR_2_example_prediction.mgf")

    # Create directories if needed.If directory already exists, does not raise exception.
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(prediction_output_folder, exist_ok=True)

    # Normalised collision energy and instrument type
    collision = 0.35
    instrument = "Lumos"

    # Parse prediction arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", default=prediction_input_path, help="Input file (.txt)", type=str)
    parser.add_argument("-output", default=prediction_output_path, help="Output file (.mgf)", type=str)
    parser.add_argument("-collision", default=collision, help="Normalised collision energy (float)", type=float)
    parser.add_argument("-instrument", default=instrument, help="Instrument used during peptide processing", type=str)
    args = parser.parse_args()

    # Command line input values used instead of defaults/customs if applicable
    prediction_input_path = args.input
    prediction_output_path = args.output
    collision = args.collision
    instrument = args.instrument

    # Model configuration
    configuration = model_configurations.Configuration_HCD_All_Modifications()

    # Get model instance
    model = rnn_model.SpecPred_Model(configuration)

    # Build model
    print("Building model...")
    # Adjust these according to the given model
    model_input_size = configuration.model_input_size
    model_output_size = configuration.get_tensorflow_output_size()
    number_layers = model.number_layers
    model.build_model(input_size=model_input_size, output_size=model_output_size, number_layers=number_layers)

    # Load model weights
    print("Loading model weights...")
    model.load_model(model_path=model_path)

    # Load input peptide data
    print("Preparing data...")
    input_peptides = load_input.read_input_file_prediction(prediction_input_path, configuration, collision=collision, instrument=instrument)

    # Begin timer
    start_time = time.perf_counter()

    # Prediction process: Generate spectral data
    print("Generating spectra...")
    output_predicted_spectra = model.predict_spectra(input_peptides)

    # Determine prediction time
    predict_time = time.perf_counter()

    # Write output file (Mascot Generic Format: .mgf)
    print("Writing output file...")
    prediction_output.write_prediction_output_mgf(prediction_output_path, input_peptides, output_predicted_spectra, configuration)

    print("Results location: ({})".format(prediction_output_folder))
    print("Prediction time: {:.3f}".format(predict_time - start_time))
    print("\n=-=-=-=-=-=-=-= Prediction Complete =-=-=-=-=-=-=-=\n")


if __name__ == "__main__":
    main()
