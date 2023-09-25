import spectraflow.resources.peptide_modifications as peptide_modifications


# Base Configuration with default and general values
class Configuration_Base(object):
    def __init__(self):

        # Get peptide modifications
        self.peptide_modifications = peptide_modifications.get_peptide_modifications()

        # Fragmentation technique and associated ion types
        self.fragmentation_method = None
        self.fragment_ion_types = None
        self.fragment_ion_terms = {'b{}': 'n', 'y{}': 'c', 'c{}': 'n', 'z{}': 'c', 'b{}-ModLoss': 'n',
                                   'y{}-ModLoss': 'c',
                                   'b{}-H2O': 'n', 'y{}-H2O': 'c', 'b{}-NH3': 'n', 'y{}-NH3': 'c'}

        # Peptide modification elements (constants)
        self.peptide_modification_elements_common = ['C', 'H', 'N', 'O', 'S', 'P']
        self.peptide_modification_elements_metal = ['Na', 'Ca', 'Fe', 'K', 'Mg', 'Cu']

        # Fixed modifications
        self.fixed_modifications = None
        self.fixed_modifications_amino_acids = None

        # Variable modifications
        self.variable_modifications = []
        self.variable_modifications_min = 0
        self.variable_modifications_max = 15

        # Standard instruments
        self.instruments = ['QE', 'Fusion', 'Lumos']
        self.instrument_limit = 8

        # Index of predicted fragment ions
        self.predicted_fragment_ion_index = None

        # Model input size
        self.model_input_size = 98

        # Default settings
        self.interval = 100
        self.charge_max = 2

    # Set the fixed modifications and create a dictionary mapping amino acids to their respective fixed modifications
    def set_fixed_modifications(self, fixed_mods):
        self.fixed_modifications = fixed_mods
        self.fixed_modifications_amino_acids = {}
        for fixed_mod in self.fixed_modifications:
            amino_acid = fixed_mod[fixed_mod.find('[') + 1:fixed_mod.find(']')]
            self.fixed_modifications_amino_acids[amino_acid] = fixed_mod

    # Set the variable modifications and their minimum/maximum allowed occurrences
    def set_variable_modifications(self, variable_modifications, variable_modifications_min,
                                   variable_modifications_max):
        self.variable_modifications = variable_modifications
        # Min allowed
        self.variable_modifications_min = variable_modifications_min
        # Max allowed
        self.variable_modifications_max = variable_modifications_max

    # Calculate the output size for the TensorFlow model based on the number of fragment ion types and charge states
    def get_tensorflow_output_size(self):
        return len(self.fragment_ion_types) * self.charge_max

    # Get the number of peptide modification features
    def num_peptide_modification_features(self):
        return len(self.peptide_modification_elements_common) + 2

    # Create a dictionary mapping fragment ion types to their corresponding indices
    def set_predicted_fragment_ion_index(self):
        self.predicted_fragment_ion_index = dict(zip(self.fragment_ion_types, range(len(self.fragment_ion_types))))

    # Set the fragment ion types and update the predicted fragment ion index
    def set_fragment_ion_types(self, fragment_ion_types):
        self.fragment_ion_types = fragment_ion_types
        self.set_predicted_fragment_ion_index()

    # Get a list of fragment ion type names without the '{}' placeholders
    def get_fragment_ion_type_names(self):
        return [fragment_ion_type.replace("{}", "") for fragment_ion_type in self.fragment_ion_types]

    # Get the fragment ion string based on the peptide sequence, fragment ion site, and type
    def get_fragment_ion_from_site(self, peptide, fragment_ion_site, fragment_ion_type):
        site = len(peptide) - fragment_ion_site if self.fragment_ion_terms[fragment_ion_type] == 'c' else fragment_ion_site
        return fragment_ion_type.format(site)

    # Get the index of a predicted fragment ion based on its type and charge
    def get_fragment_ion_index_from_type(self, fragment_ion_type, fragment_ion_charge):
        if fragment_ion_charge > self.charge_max or fragment_ion_type not in self.predicted_fragment_ion_index:
            return None
        return self.predicted_fragment_ion_index[fragment_ion_type] * self.charge_max + fragment_ion_charge - 1


# Custom configuration(s)

# Configuration for CCLE peptides
class Configuration_Custom_CCLE(Configuration_Base):
    def __init__(self):
        super(self.__class__, self).__init__()

        self.fragmentation_method = 'CID'
        self.set_fragment_ion_types(['b{}', 'y{}'])
        self.set_fixed_modifications(["Oxidation[M]"])
        self.set_variable_modifications(["Carbamidomethyl[C]", "TMT6plex[K]", "TMT6plex[AnyN-term]"], self.variable_modifications_min, self.variable_modifications_max)


# Configurations for main fragmentation methods

# Unmodified, no fixed or variable modifications
class Configuration_CID_Unmodified(Configuration_Base):
    def __init__(self):
        super(self.__class__, self).__init__()

        self.fragmentation_method = 'CID'
        self.set_fragment_ion_types(['b{}', 'y{}'])


class Configuration_HCD_Unmodified(Configuration_Base):
    def __init__(self):
        super(self.__class__, self).__init__()

        self.fragmentation_method = 'HCD'
        self.set_fragment_ion_types(['b{}', 'y{}'])


class Configuration_ETD_Unmodified(Configuration_Base):
    def __init__(self):
        super(self.__class__, self).__init__()

        self.fragmentation_method = 'ETD'
        self.set_fragment_ion_types(['c{}', 'z{}'])


# Configurations supporting all peptide modifications

# Generic Configuration: CID supporting all supported peptide modifications
class Configuration_CID_All_Modifications(Configuration_Base):
    def __init__(self):
        super(self.__class__, self).__init__()

        self.fragmentation_method = 'CID'
        self.set_fragment_ion_types(['b{}', 'y{}'])
        self.set_fixed_modifications(["Oxidation[M]"])
        self.set_variable_modifications(peptide_modifications.get_peptide_modification_names(), self.variable_modifications_min, self.variable_modifications_max)


# Generic Configuration: HCD supporting all supported peptide modifications
class Configuration_HCD_All_Modifications(Configuration_Base):
    def __init__(self):
        super(self.__class__, self).__init__()

        self.fragmentation_method = 'HCD'
        self.set_fragment_ion_types(['b{}', 'y{}'])
        self.set_fixed_modifications(["Oxidation[M]"])
        self.set_variable_modifications(peptide_modifications.get_peptide_modification_names(), self.variable_modifications_min, self.variable_modifications_max)


# Generic Configuration: ETD supporting all supported peptide modifications
class Configuration_ETD_All_Modifications(Configuration_Base):
    def __init__(self):
        super(self.__class__, self).__init__()

        self.fragmentation_method = 'ETD'
        self.set_fragment_ion_types(['c{}', 'z{}'])
        self.set_fixed_modifications(["Oxidation[M]"])
        self.set_variable_modifications(peptide_modifications.get_peptide_modification_names(), self.variable_modifications_min, self.variable_modifications_max)
