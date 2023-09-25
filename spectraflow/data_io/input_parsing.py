import numpy as np


# Convert input data dictionaries to numpy
def convert_to_numpy(data):

    converted_data = {}

    # Iterate over each peptide length and its corresponding value
    for peptide_length, value in data.items():
        converted_list = []

        # Get the length of the value list
        value_length = len(value)

        # Iterate over the indices of the first value in the current item's value list
        for i in range(len(value[0])):
            current_values = [value[j][i] for j in range(value_length)]
            current_value_type = type(current_values[0])

            # Check if the current values are NumPy arrays
            if isinstance(current_values[0], np.ndarray):
                # Stack the arrays along a new axis
                x = np.stack(current_values)
            else:
                # Infer the data type based on the current value
                if current_value_type == str:
                    dtype = None
                elif current_value_type == float:
                    dtype = np.float32
                else:
                    dtype = np.int8

                # Create a NumPy array from the values with the specified data type
                x = np.array(current_values, dtype=dtype)

            converted_list.append(x)

        # Store the converted list for the current peptide length
        converted_data[peptide_length] = converted_list

    return converted_data


class ExtractPeptideInput:
    def __init__(self, configuration):

        # Configuration
        self.configuration = configuration

        # Instrument
        self.instrument_feature = None
        self.process_instrument()

        # Sample limit
        self.sample_limit = 100000000

        # Peptide sequences/amino acids
        # Valid AA symbols
        self.valid_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        # Encode amino acids into one hot vectors
        self.encoded_amino_acids = self.encode_amino_acids()
        self.amino_acid_index = {aa: index for index, aa in enumerate(self.valid_amino_acids)}

        # Peptide modifications
        self.modification_feature = None
        self.num_peptide_modification_features = None
        self.peptide_modification_elements_common = self.configuration.peptide_modification_elements_common
        self.peptide_modification_elements_metal = self.configuration.peptide_modification_elements_metal
        self.process_modifications()

        # Fixed modifications
        self.fixed_modification_amino_acids = {}
        if self.configuration.fixed_modifications is not None:
            for fixed_peptide_mod in self.configuration.fixed_modifications:
                amino_acid = fixed_peptide_mod[fixed_peptide_mod.find('[') + 1:fixed_peptide_mod.find(']')]
                self.fixed_modification_amino_acids[amino_acid] = fixed_peptide_mod

    def process_instrument(self):

        # Initialize instrument_feature dictionary
        self.instrument_feature = {}

        # Get instrument limit and instrument list from configuration
        instrument_limit = self.configuration.instrument_limit
        instruments = self.configuration.instruments

        # Create a 2D identity matrix
        feature = np.eye(instrument_limit, dtype=np.int8)

        # Assign instrument features to instrument_feature dictionary
        for i in range(len(instruments)):
            # Lowercase the instrument name
            self.instrument_feature[instruments[i].lower()] = feature[i]

        # Assign 'unknown' feature to instrument_feature dictionary
        self.instrument_feature['unknown'] = feature[-1]

    def process_modifications(self):
        # Get the number of peptide modification features from the configuration
        self.num_peptide_modification_features = self.configuration.num_peptide_modification_features()
        # Initialize an empty dictionary to store modification features
        self.modification_feature = {}

        def process_modification_properties(properties_input):
            # Create a list to store the feature values for the modification
            feature = [0] * self.num_peptide_modification_features
            # Split the properties input into individual property strings
            properties_split = properties_input.split(')')[:-1]

            # Iterate over each modification property string
            for modification_property in properties_split:
                # Split the property into chemical and n (number of occurrences)
                chemical, n = modification_property.split('(')
                n = int(n)

                try:
                    # Find the index of the chemical in the common modification elements
                    index = self.peptide_modification_elements_common.index(chemical)
                    # Update the corresponding feature value
                    feature[index] = n
                except ValueError:
                    # If the chemical is not found in the common modification elements
                    if chemical in self.peptide_modification_elements_metal:
                        feature[-2] += n
                    # Update the feature value for metal elements
                    else:
                        # Update the last feature value for other elements
                        feature[-1] += n

            return feature

        # Process modifications for each modification name and properties in the configuration
        for modification_name, modification_properties in self.configuration.peptide_modifications.items():
            # Extract the properties
            properties = modification_properties.split(' ')[-1]
            # Process the modification properties and store the resulting feature array
            self.modification_feature[modification_name] = np.array(process_modification_properties(properties), dtype=np.int8)

    def encode_amino_acids(self):
        # Mapping amino acid symbols to one hot vectors
        encoded_amino_acids_map = {
            symbol: np.eye(len(self.valid_amino_acids), dtype=int)[i]
            for i, symbol in enumerate(self.valid_amino_acids)
        }
        return encoded_amino_acids_map

    def parse_peptide_vector(self, peptide, fragment_ion_index):

        vector = []
        sequence_index = fragment_ion_index

        # Read the ion's previous N-terminus amino acid (sequence id -1)
        vector.extend(self.encoded_amino_acids.get(peptide[sequence_index - 1], [0] * len(self.encoded_amino_acids)))

        # Read the ion's next C-terminus amino acid (sequence id +1)
        vector.extend(self.encoded_amino_acids.get(peptide[sequence_index], [0] * len(self.encoded_amino_acids)))

        # Number of amino acids by terminus (N, C)
        terminus_n_amino_acids = [0] * len(self.encoded_amino_acids)
        terminus_c_amino_acids = [0] * len(self.encoded_amino_acids)

        # Calculate amino acid counts for N terminus and C terminus
        for i, amino_acid in enumerate(peptide):
            if amino_acid in self.amino_acid_index:
                if i < sequence_index - 1:
                    terminus_n_amino_acids[self.amino_acid_index[amino_acid]] += 1
                elif i > sequence_index + 1:
                    terminus_c_amino_acids[self.amino_acid_index[amino_acid]] += 1

        vector.extend(terminus_n_amino_acids)
        vector.extend(terminus_c_amino_acids)

        terminus_n = int(fragment_ion_index == 1)
        terminus_c = int(fragment_ion_index == len(peptide) - 1)
        vector.extend([terminus_n, terminus_c])

        return np.array(vector, dtype=np.int8)

    # Featurise peptide
    def featurise_peptide(self, peptide, modification_properties):
        # Check peptide length
        if len(peptide) > self.configuration.interval + 1:
            return None

        # Initialize modification feature index
        modification_feature_index = np.zeros((len(peptide), self.num_peptide_modification_features), dtype=np.int8).tolist()

        # Initialize variables
        modifications = []
        variable_modifications = 0
        modification_unexpected = False
        modification_properties_split = modification_properties.split(';')

        # Adjust the feature index based on the peptide length
        def index_feature(index, peptide_length):
            index = min(max(index - 1, 0), peptide_length - 1)
            return index

        # Process each modification
        for modification in modification_properties_split:
            if not modification:
                continue

            # Split modification into index and name
            split_mod = modification.split(',')
            index = int(split_mod[0])
            modification_name = split_mod[1]
            modifications.append((index, modification_name))

            # Handle fixed modifications
            if modification_name in self.configuration.fixed_modifications:
                index = index_feature(index, len(peptide))
                modification_feature_index[index] = self.modification_feature[modification_name]

            # Handle variable modifications
            elif modification_name in self.configuration.variable_modifications:
                index = index_feature(index, len(peptide))
                modification_feature_index[index] = self.modification_feature[modification_name]
                variable_modifications += 1

            # Handle unexpected modifications
            else:
                modification_unexpected = True
                break

        # Validation measures
        # Invalid number of modifications
        if variable_modifications < self.configuration.variable_modifications_min or variable_modifications > self.configuration.variable_modifications_max:
            print(f"Peptide {peptide} excluded for invalid number of modifications: {variable_modifications}")
            return None
        # Unexpected modification
        if modification_unexpected:
            print(f"Peptide {peptide} excluded for containing an unexpected modification: {modification_name}")
            return None
        # Invalid amino acid symbol
        if any(amino_acid not in self.encoded_amino_acids for amino_acid in peptide):
            print(f"Peptide {peptide} excluded for containing an invalid amino acid")
            return None

        # Generate features
        x = []
        modification_x = []
        for site in range(1, len(peptide)):
            modification_x.append(np.append(modification_feature_index[site - 1], modification_feature_index[site]))
            x.append(self.parse_peptide_vector(peptide, site))
        x, modification_x = np.array(x), np.array(modification_x)

        return x, modification_x

    def featurise_input_training(self, input_file_training, collision, instrument):
        with open(input_file_training) as f:
            normalised_instrument = self.instrument_feature.get(instrument.lower(), self.instrument_feature['unknown'])

            grouped_peptides = {}
            header_line = f.readline().strip()
            header_index = dict(zip(header_line.split('\t'), range(len(header_line))))

            # Check if the charge is specified in the header
            charge_i_s = "charge" not in header_index

            # Counter for the number of processed samples
            num_samples = 0

            # Iterate over each line in the file
            for line in f:
                split_values = line.split("\t")
                peptide = split_values[1]

                # Determine the precursor charge
                # Check if the charge is specified in the header as a column
                if "charge" not in header_index:
                    precursor = int(split_values[0].split(".")[-3])
                else:
                    # Otherwise, extract from spec column
                    precursor = int(split_values[header_index["charge"]])

                modification_properties = split_values[2]

                # Featurise the peptide and modification properties
                featurised_data = self.featurise_peptide(peptide, modification_properties)

                # Skip the current peptide if feature conversion is not possible
                if featurised_data is None:
                    continue

                ion_type_intensity_map = {}
                # Process each fragment ion type
                for fragment_ion_type in self.configuration.get_fragment_ion_type_names():
                    peaks = split_values[header_index[fragment_ion_type]]
                    if len(peaks) < 2:
                        continue
                    peaks = [peak.split(",") for peak in peaks.strip().strip(";").split(";")]
                    ion_type_intensity_map.update(dict([(peak[0], float(peak[1])) for peak in peaks]))

                # Skip the current peptide if the number of fragment ions is insufficient
                if len(ion_type_intensity_map) < len(peptide):
                    continue

                intensities = []
                # Iterate over each site in the peptide
                for site in range(1, len(peptide)):

                    feature_vector = []

                    # Process each fragment ion type for the current site and charge
                    for fragment_ion_type in self.configuration.fragment_ion_types:
                        fragment_ion_name = self.configuration.get_fragment_ion_from_site(peptide, site, fragment_ion_type)
                        feature_vector.extend([ion_type_intensity_map.get(ion_name_charge, 0) for charge in range(1, self.configuration.charge_max + 1) for ion_name_charge in [fragment_ion_name + "+{}".format(charge)]])
                    intensities.append(np.array(feature_vector, dtype=np.float32))

                intensities = np.array(intensities)
                intensities /= np.max(intensities)
                peptide_length = len(peptide)

                # Group the peptides by length and store the relevant information
                grouped_peptides.setdefault(peptide_length, []).append(
                    (featurised_data[0], featurised_data[1], precursor, float(collision), normalised_instrument, intensities))

                num_samples += 1
                # Check if the sample limit has been reached
                if num_samples >= self.sample_limit:
                    break
        # Convert the grouped peptides dictionary to numpy
        return convert_to_numpy(grouped_peptides)

    def featurise_input_prediction(self, input_peptides, collision, instrument):

        # Determine the instrument feature based on the provided instrument
        instrument_feature = self.instrument_feature.get(instrument.lower(), self.instrument_feature['unknown'])

        # Dictionary to store grouped peptides by length
        grouped_peptides = {}

        # Iterate over each peptide in the peptide list
        for peptide, modification_properties, precursor in input_peptides:
            precursor = int(precursor)

            # Featurise the peptide and modification properties
            featurised_data = self.featurise_peptide(peptide, modification_properties)

            # Skip the current peptide if featurisation is not possible
            if featurised_data is None:
                continue

            peptide_length = len(peptide)
            peptide_properties = f"{peptide}|{modification_properties}|{precursor}"

            # Group the peptides by length and store
            if peptide_length in grouped_peptides:
                grouped_peptides[peptide_length].append((featurised_data[0], featurised_data[1], precursor, float(collision), instrument_feature, peptide_properties))
            else:
                grouped_peptides[peptide_length] = [(featurised_data[0], featurised_data[1], precursor, float(collision), instrument_feature, peptide_properties)]

        # Convert the grouped peptides dictionary to numpy
        return convert_to_numpy(grouped_peptides)
