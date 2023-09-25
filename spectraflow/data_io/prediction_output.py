import numpy as np
import spectraflow.processing.fragment_ion_processing as fragment_ion_processing


# Write MGF file detailing spectral predictions for the input peptides
def write_prediction_output_mgf(output_file, input_peptides, output_predicted_spectra, configuration):
    with open(output_file, 'w') as f:
        for key, value in input_peptides.items():
            predicted_spectra = output_predicted_spectra[key][-1]
            for i in range(value[-1].shape[0]):
                write_single_peptide(f, value[-1][i], predicted_spectra[i], configuration)


# Peptide by peptide
def write_single_peptide(f, peptide_data, prediction_data, configuration):
    # Split peptide_data into peptide, mod, and charge
    peptide_sequence, modification, charge = peptide_data.split("|")
    f.write('BEGIN IONS\n')
    f.write('TITLE=' + peptide_data + '\n')
    f.write('CHARGE=' + charge + '\n')
    precursor = int(charge)
    f.write('pepinfo=' + peptide_data + '\n')

    # Calculate fragment ions
    fragment_ions = calculate_fragment_ions(peptide_sequence, modification, configuration)

    peaks = []
    charge_max = min(precursor, configuration.charge_max)

    # Iterate over fragment ion types
    for fragment_ion_type in fragment_ions:
        ions = np.array(fragment_ions[fragment_ion_type])
        for charge in range(1, charge_max + 1):

            # Get intensities for the current fragment ion type and charge
            intensities = prediction_data[:, configuration.get_fragment_ion_index_from_type(fragment_ion_type, charge)]
            f.write('{}={}\n'.format(fragment_ion_type.format("+" + str(charge)), ','.join('%.5f' % intensity for intensity in intensities)))
            # Types for each site in the peptide
            types = [configuration.get_fragment_ion_from_site(peptide_sequence, site, fragment_ion_type) + "+" + str(charge) for site in range(1, len(peptide_sequence))]
            peaks.extend(zip(ions / charge + fragment_ion_processing.masses.mass_proton, intensities, types))

    # Calculate mass
    peptide_mass = determine_peptide_mass(peptide_sequence, modification, precursor)
    f.write("PEPMASS=%.5f\n" % peptide_mass)

    # Sort and write peak_list to the file
    peaks.sort()
    for mass_charge_mz, intensity, fragment_ion_type in peaks:
        if intensity > 1e-8:
            f.write("%f %.8f %s\n" % (mass_charge_mz, intensity, fragment_ion_type))
    f.write('END IONS\n')


# Fragment ions
def calculate_fragment_ions(peptide, mod, configuration):
    fragment_ions = {}
    # Calculate b ions and peptide mass
    ions_b, peptide_mass = fragment_ion_processing.determine_b_ions(peptide, mod)

    fragment_ion_types = configuration.fragment_ion_types

    # Types of fragment ions
    # b-ions
    if 'b{}' in fragment_ion_types:
        fragment_ions['b{}'] = ions_b
    # y-ions
    if 'y{}' in fragment_ion_types:
        fragment_ions['y{}'] = fragment_ion_processing.determine_y_ions(ions_b, peptide_mass)
    # c-ions
    if 'c{}' in fragment_ion_types:
        fragment_ions['c{}'] = fragment_ion_processing.determine_c_ions(ions_b)
    # z-ions
    if 'z{}' in fragment_ion_types:
        fragment_ions['z{}'] = fragment_ion_processing.determine_z_ions(ions_b, peptide_mass)

    return fragment_ions


# Peptide mass
def determine_peptide_mass(peptide_sequence, modification, charge):
    ions_b, peptide_mass = fragment_ion_processing.determine_b_ions(peptide_sequence, modification)
    peptide_mass = peptide_mass / charge + fragment_ion_processing.masses.mass_proton
    return peptide_mass
