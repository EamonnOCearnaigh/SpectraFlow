import os
import time

import spectraflow.rnn.model_configurations as model_configurations
import spectraflow.rnn.rnn_model as rnn_model
import spectraflow.processing.peptide_processing as peptide_processing
import spectraflow.data_io.load_input_train_test as load_input_train_test


def main():

    print("\n=-=-=-=-=-=-=-= Training Beginning =-=-=-=-=-=-=-=\n")

    # Path management for portability
    # Get the directory containing script
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct paths to the data and model folders
    data_folder_path = os.path.join(script_directory, 'data')
    model_folder_path = os.path.join(script_directory, 'models')

    # Path for input training data: plabel files (by folder)
    # Can extend to multiple folders, to be combined below
    #training_input_folder = os.path.join(data_folder_path, "input_training", "ICR_1")
    training_input_folder = os.path.join(data_folder_path, "input_training", "ICR_2")
    # etc.

    # Path for output model
    #model_folder = os.path.join(model_folder_path, "ICR_1")
    model_folder = os.path.join(model_folder_path, "ICR_2")
    model_path = os.path.join(model_folder, "baseline_model_weights.h5")

    # Create directories if needed.If directory already exists, does not raise exception.
    os.makedirs(model_folder, exist_ok=True)

    # Get model configuration
    configuration = model_configurations.Configuration_HCD_All_Modifications()

    # Get rnn model
    model = rnn_model.SpecPred_Model(configuration)

    # Model building settings
    model.input_size = configuration.model_input_size
    model.output_size = configuration.get_tensorflow_output_size()

    # Model training settings
    epochs = [15, 20, 25, 50, 75, 100, 125, 150]
    neurons_per_layer = [128, 256, 512]
    batch_sizes = [256, 512, 1024, 2048]
    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    layers = [2, 3, 4]
    loss_function = 'mean_absolute_error'

    # Training: additional input
    collision = 0.35
    instrument = 'Lumos'
    sample_limit = 100000000

    # Parameter selections
    model.epochs = epochs[0]
    model.number_layers = layers[1]
    model.layer_size = neurons_per_layer[1]
    model.batch_size = batch_sizes[2]
    model.learning_rate = learning_rates[0]

    # Build tensorflow model
    print("Building model...")
    model.build_model(input_size=model.input_size, output_size=model.output_size, number_layers=model.number_layers)

    # Compile model
    print("Compiling model...")
    model.model.compile(loss=loss_function, optimizer=model.optimiser)

    # Grouping input peptide data
    # Can combine multiple folders
    peptides = {}
    grouped_peptides = peptide_processing.combine_groups(peptides, load_input_train_test.read_plabel_input_directory(
        training_input_folder, configuration, collision=collision, instrument=instrument, sample_limit=sample_limit))

    # Print training input data, grouped by mass
    printer = peptide_processing.PeptideGroupsPrinter(grouped_peptides)
    printer.print_groups()

    # Start timer
    print("Training...")
    start_time = time.perf_counter()

    # Training model
    model.train_model(grouped_peptides, save_path=model_path)

    # Determine training time
    train_time = time.perf_counter()

    print("Trained model location: ({})".format(model_folder))
    print("Training time: {:.3f}".format(train_time - start_time))
    print("\n=-=-=-=-=-=-=-= Training Complete =-=-=-=-=-=-=-=\n")


if __name__ == '__main__':
    main()