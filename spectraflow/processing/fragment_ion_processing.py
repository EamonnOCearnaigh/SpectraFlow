import spectraflow.resources.masses as masses

# Get masses
masses = masses.Masses()


def modification_masses(peptide_sequence, modification_properties):
    # Split the modification properties and construct a list of modifications with their corresponding sites and names
    modifications = [(int(modification.split(",")[0]), modification.split(",")[1]) for modification in modification_properties.split(";") if modification]
    # Sort the modifications based on the modification sites
    modifications.sort()

    modification_mass = [0] * (len(peptide_sequence) + 2)
    loss_mass = [0] * (len(peptide_sequence) + 2)
    modification_name = [""] * (len(peptide_sequence) + 2)

    for modification in modifications:
        # Retrieve the modification mass and loss mass from the masses object using the modification name
        mod_mass, mod_loss = masses.peptide_modification_masses.get(modification[1], (0, 0))
        modification_mass[modification[0]] = mod_mass
        loss_mass[modification[0]] = mod_loss

        # Priority values for modifications that can cause a loss of mass
        if modification[1] in {"Oxidation[M]": 1e5, "Phospho[T]": 1e7, "Phospho[S]": 1e8, "": 0}:
            modification_name[modification[0]] = modification[1]

    return modification_mass, loss_mass, modification_name


# Determining fragment ions: b, y, c, z
def determine_b_ions(peptide_sequence, modification_properties):

    b_ions = []
    mod_masses, _, _ = modification_masses(peptide_sequence, modification_properties)
    n_terminus_mass = mod_masses[0]

    # Calculate the mass of the N-terminus fragment ion (b-ion) by accumulating the mass of each amino acid and modification
    for i, amino_acid in enumerate(peptide_sequence[:-1]):
        n_terminus_mass += masses.amino_acid_masses.get(amino_acid, 0) + mod_masses[i + 1]
        b_ions.append(n_terminus_mass)

    # Calculate the total peptide mass by summing the mass of the last b-ion, the last amino acid, and the modification masses
    peptide_mass = (
            b_ions[-1]
            + masses.amino_acid_masses.get(peptide_sequence[-1], 0)
            + mod_masses[len(peptide_sequence)]
            + mod_masses[len(peptide_sequence) + 1]
            + masses.mass_H2O
    )
    return b_ions, peptide_mass


# Calculate the y-ions by subtracting each b-ion mass from the total peptide mass
def determine_y_ions(b_ions, peptide_mass):
    return [peptide_mass - b_ion for b_ion in b_ions]


# Calculate the c-ions by adding the mass of NH3 to each b-ion mass
def determine_c_ions(b_ions):
    return [b_ion + masses.mass_NH3 for b_ion in b_ions]


# Calculate the z-ions by subtracting each b-ion mass and the mass of NH3
# from the total peptide mass and adding the mass of H
def determine_z_ions(b_ions, peptide_mass):
    return [peptide_mass - b_ion - masses.mass_NH3 + masses.mass_H for b_ion in b_ions]
