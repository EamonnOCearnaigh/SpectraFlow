from spectraflow import predict_spectra
from spectraflow import train_model
from spectraflow import test_model


def main():
    # This script is to demonstrate the three main scripts of SpectraFlow,
    # showing training, testing and prediction using a custom TensorFlow BiLSTM model.

    # Please install:
    "pip install spectraflow-1.0.0.tar.gz"
    # The file can be found in dist/

    # Testing new model
    test_model.main()

    # Predicting spectra using trained model
    predict_spectra.main()

    # Training new model
    train_model.main()


if __name__ == '__main__':
    main()
