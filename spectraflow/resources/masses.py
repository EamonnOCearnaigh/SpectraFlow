import spectraflow.resources.peptide_modifications as peptide_modifications


# Class containing mass constants for amino acids, elements, compounds

class Masses(object):

    #  Initialise masses class
    def __init__(self):
        # Amino acid masses (in order of symbol)
        self.amino_acid_masses = {'A': 71.037114, 'B': 0., 'C': 103.009185, 'D': 115.026943, 'E': 129.042593,
                                  'F': 147.068414, 'G': 57.021464, 'H': 137.058912, 'I': 113.084064, 'J': 114.042927,
                                  'K': 128.094963, 'L': 113.084064, 'M': 131.040485, 'N': 114.042927, 'P': 97.052764,
                                  'Q': 128.058578, 'R': 156.101111, 'S': 87.032028, 'T': 101.047679, 'U': 150.95363,
                                  'V': 99.068414, 'X': 0., 'W': 186.079313, 'Y': 163.06332, 'Z': 0.}

        # Glyco masses (in order of mass)
        self.glyco_masses = {"Xyl": 132.0422587452, "Hex": 162.0528234315, "dHex": 146.0579088094,
                             "HexNAc": 203.07937253300003, "NeuAc": 291.09541652769997, "NeuGc": 307.09033114979997}

        # Proton mass
        self.mass_proton = 1.007276

        # Elemental masses
        # Hydrogen
        self.mass_H = 1.0078250321
        # Nitrogen
        self.mass_N = 14.0030740052
        # Carbon
        self.mass_C = 12.00
        # Oxygen
        self.mass_O = 15.9949146221

        # Compound masses calculations using elemental masses
        # Carbon Monoxide
        self.mass_CO = self.mass_C + self.mass_O
        # Carbon Dioxide
        self.mass_CO2 = self.mass_C + self.mass_O * 2
        # Water
        self.mass_H2O = self.mass_H * 2 + self.mass_O
        # Hydroxyl
        self.mass_HO = self.mass_H + self.mass_O
        # Imidogen
        self.mass_NH = self.mass_N + self.mass_H
        # Ammonia
        self.mass_NH3 = self.mass_N + self.mass_H * 3

        # N & C Terminus
        self.mass_n_terminus = self.mass_H
        self.mass_c_terminus = self.mass_O + self.mass_H

        # Peptide modification masses
        self.peptide_modification_masses = peptide_modifications.get_peptide_modification_masses()
