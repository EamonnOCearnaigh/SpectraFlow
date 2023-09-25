import os

import numpy as np
import tensorflow as tf

import spectraflow.processing.peptide_processing as peptide_processing
from spectraflow.rnn import model_configurations

# Set random seed for tensorflow operations
tf.random.set_seed(2023)


class SpecPred_Model(tf.keras.Model):
    def __init__(self, configuration):

        super(SpecPred_Model, self).__init__()

        # Default settings
        self.model = None
        self.configuration = configuration
        self.epochs = 2
        self.batch_size = 1024
        self.number_layers = 3
        self.neurons_per_layer = 256
        self.learning_rate = 0.01
        self.activation_func = 'tanh'
        self.optimiser = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.keep_probability = 0.8  # Dropout

    def get_config(self):
        config = super(SpecPred_Model, self).get_config()
        config.update({
            "configuration": self.configuration,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "number_layers": self.number_layers,
            "neurons_per_layer": self.neurons_per_layer,
            "learning_rate": self.learning_rate,
            "activation_func": self.activation_func,
            "optimiser": self.optimiser,
            "keep_probability": self.keep_probability
        })
        return config

    # Building
    def build_model(self, input_size, output_size, number_layers):

        _x = tf.keras.layers.Input(shape=(None, input_size), name="input_x")
        _charge = tf.keras.layers.Input(shape=(None, 1), name="input_charge")
        _time_step = tf.keras.layers.Input(shape=(), dtype=tf.int32, name="input_time_step")
        _nce = tf.keras.layers.Input(shape=(None, 1), name="input_nce")
        _instrument = tf.keras.layers.Input(shape=(None, self.configuration.instrument_limit), name="input_instrument")
        _y = tf.keras.layers.Input(shape=(None, output_size), name="input_y")

        x = tf.concat([_x, _charge], axis=2)
        instrument = tf.concat([_instrument, _nce], axis=2)
        instrument = tf.keras.layers.Dense(3)(instrument)
        x = tf.concat([x, instrument], axis=2)

        for _ in range(number_layers):
            lstm_fw_cell = tf.keras.layers.LSTM(self.neurons_per_layer, activation=self.activation_func,
                                                return_sequences=True)
            lstm_bw_cell = tf.keras.layers.LSTM(self.neurons_per_layer, activation=self.activation_func,
                                                return_sequences=True)
            x_fw = lstm_fw_cell(x)
            x_bw = lstm_bw_cell(x)
            x_fw = tf.keras.layers.Dropout(rate=1 - tf.squeeze(self.keep_probability))(x_fw)
            x_bw = tf.keras.layers.Dropout(rate=1 - tf.squeeze(self.keep_probability))(x_bw)
            x = tf.concat([x_fw, x_bw], axis=2)
            x = tf.keras.layers.Dropout(rate=1 - tf.squeeze(self.keep_probability))(x)
            x = tf.concat([x, _charge], axis=2)
            x = tf.concat([x, instrument], axis=2)

        # Output layer
        output_layer = tf.keras.layers.Dense(output_size)
        outputs = output_layer(x)

        self.model = tf.keras.Model(
            inputs=[_x, _charge, _time_step, _nce, _instrument, _y],
            outputs=outputs)

    def convert_value_for_tf(self, ch, time_step, outsize=1):
        ch = tf.reshape(ch, (-1, 1, outsize))
        ch = tf.repeat(ch, time_step, axis=1)
        ch = tf.cast(ch, tf.float32)
        return ch

    def mod_feature(self, x, mod_x):
        return tf.concat((x, mod_x), axis=2)

    # Training
    def train_model(self, grouped_peptides, save_path=None):
        peptide_batch = peptide_processing.Grouped_Peptide_Batch(grouped_peptides, batch_size=self.batch_size, batch_shuffle=True)

        mean_costs = []

        for epoch in range(self.epochs):
            batch_cost = []
            batch_time_cost = []
            ith_batch = 0
            peptide_batch.reset_batch()

            while True:
                batch = peptide_batch.get_next_batch()
                if batch is None:
                    break
                ith_batch += 1

                peplen = peptide_batch.get_data_from_batch(batch, "peptide_length")
                ch = np.float32(peptide_batch.get_data_from_batch(batch, "charge"))
                x = np.float32(peptide_batch.get_data_from_batch(batch, "x"))
                mod_x = np.float32(peptide_batch.get_data_from_batch(batch, "mod_x"))
                instrument = np.float32(peptide_batch.get_data_from_batch(batch, "instrument"))
                nce = np.float32(peptide_batch.get_data_from_batch(batch, "collision"))
                y = peptide_batch.get_data_from_batch(batch, "y")

                x = self.mod_feature(x, mod_x)
                ch = self.convert_value_for_tf(ch, peplen[0] - 1)
                nce = self.convert_value_for_tf(nce, peplen[0] - 1)
                instrument = self.convert_value_for_tf(instrument, peplen[0] - 1, instrument.shape[-1])

                with tf.GradientTape() as tape:
                    predictions = self.model([x, ch, peplen - 1, nce, instrument, y], training=True)
                    loss = tf.reduce_mean(tf.abs(predictions - y))

                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimiser.apply_gradients(zip(grads, self.model.trainable_variables))

                batch_cost.append(loss.numpy())
                batch_time_cost.append(0)

            mean_costs.append(np.mean(batch_cost))
            print(f"Epoch: {epoch + 1} | Mean Cost: {mean_costs[-1]}")

        if save_path:
            self.save_model(save_path)

    # Model Prediction
    def predict_spectra(self, grouped_peptides):

        peptide_batch = peptide_processing.Grouped_Peptide_Batch(grouped_peptides, batch_size=self.batch_size, batch_shuffle=False)
        output_grouped_peptides = {}

        while True:
            batch = peptide_batch.get_next_batch()
            if batch is None:
                break

            peplen = peptide_batch.get_data_from_batch(batch, "peptide_length")
            charge = np.float32(peptide_batch.get_data_from_batch(batch, "charge"))
            x = np.float32(peptide_batch.get_data_from_batch(batch, "x"))
            mod_x = np.float32(peptide_batch.get_data_from_batch(batch, "mod_x"))
            instrument = np.float32(peptide_batch.get_data_from_batch(batch, "instrument"))
            nce = np.float32(peptide_batch.get_data_from_batch(batch, "collision"))
            # Target (y) excluded

            x = self.mod_feature(x, mod_x)
            charge = self.convert_value_for_tf(charge, peplen[0] - 1)
            nce = self.convert_value_for_tf(nce, peplen[0] - 1)
            instrument = self.convert_value_for_tf(instrument, peplen[0] - 1, instrument.shape[-1])
            time_step = peplen - 1

            # Create a dummy target placeholder array
            configuration = model_configurations.Configuration_CID_All_Modifications()
            output_size = configuration.get_tensorflow_output_size()
            dummy_target = np.zeros((x.shape[0], x.shape[1], output_size))

            # Generate predictions
            predictions = self.model.predict([x, charge, time_step, nce, instrument, dummy_target])
            predictions[predictions > 1] = 1
            predictions[predictions < 0] = 0
            _grouped_peptides = {peplen[0]: (predictions,)}
            output_grouped_peptides = peptide_processing.combine_groups(output_grouped_peptides, _grouped_peptides)

        return output_grouped_peptides

    # Save trained tensorflow model
    def save_model(self, save_path):

        # Save model weights (.h5)
        self.model.save_weights(os.path.join(save_path))

    # Load trained tensorflow model for prediction/testing
    def load_model(self, model_path):

        # Load model weights (.h5)
        self.model.load_weights(model_path)
