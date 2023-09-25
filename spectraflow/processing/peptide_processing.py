import numpy as np


# Print peptide mass groups from plabel files (training, testing)
class PeptideGroupsPrinter:
    def __init__(self, groups):
        self.groups = groups

    # As above, includes quantities and shapes
    def print_groups(self):
        total_size = 0
        print("\nPeptide data grouped by mass:")
        for peptide_length, data in self.groups.items():
            group_shapes = [f'x{i} shape: {value.shape} ' for i, value in enumerate(data)]
            print(f"Peptide length: {peptide_length}\n{''.join(group_shapes)}")
            total_size += data[0].shape[0]
        # All data
        print(f"Total quantity of grouped data: {total_size}\n")


# Combine groups of peptides together
def combine_groups(groups, additional_groups):
    combined_groups = groups.copy()

    def combine_groups_tuples(tuple_a, tuple_b):
        return [np.append(x, y, axis=0) for x, y in zip(tuple_a, tuple_b)]

    for key, value in additional_groups.items():
        if key in combined_groups:
            combined_groups[key] = combine_groups_tuples(combined_groups[key], value)
        else:
            combined_groups[key] = value
    return combined_groups


# Batching peptides for model training and testing
# Derived from pDeep2
class Grouped_Peptide_Batch(object):
    def __init__(self, groups, batch_size, batch_shuffle):

        self.groups = groups
        self.batch_size = batch_size
        self.batch_shuffle = batch_shuffle
        self.group_keys = np.array(list(groups.keys()), dtype=np.int32)

        # Reset
        self.reset_batch()

        if len(groups) != 0:
            first_key = next(iter(groups))
            self.tuple_len = len(groups[first_key])
        else:
            self.tuple_len = 0

        self.features = ['x', 'mod_x', 'charge', 'collision', 'instrument', 'y']
        self.index_tuple = dict(zip(self.features, range(len(self.features))))
        self.index_tuple['peptide_length'] = -1

    # Retrieve data
    def get_data_from_batch(self, batch, name):
        return batch[self.index_tuple[name]]

    # Reset
    def reset_batch(self):
        if self.batch_shuffle:
            np.random.shuffle(self.group_keys)
        self.current_key = 0
        self.reset_current_group()

    # Reset group
    def reset_current_group(self):
        if self.current_key < len(self.group_keys):
            self.current_group_indices = np.arange(len(self.groups[self.group_keys[self.current_key]][0]))
            if self.batch_shuffle:
                self.current_group_indices = np.random.permutation(self.current_group_indices)
            self.reset_current_indices()

    # Reset indices
    def reset_current_indices(self):
        self.current_batch_start = 0
        self.current_batch_end = self.batch_size
        if self.current_batch_end > len(self.current_group_indices):
            self.current_batch_end = len(self.current_group_indices)

    # next batch
    def get_next_batch(self):

        if self.current_key >= len(self.group_keys):
            return None
        peptide_length = self.group_keys[self.current_key]

        # Single batch
        def get_one_batch(group_value):

            batch = []
            for i in range(len(group_value)):
                batch.append(
                    group_value[i][self.current_group_indices[self.current_batch_start:self.current_batch_end]])
            return batch

        batch = get_one_batch(self.groups[peptide_length])
        batch.append(np.array([peptide_length] * (self.current_batch_end - self.current_batch_start), dtype=np.int32))

        if self.current_batch_end == len(self.current_group_indices):
            self.current_key += 1
            self.reset_current_group()

        else:
            self.current_batch_start = self.current_batch_end
            self.current_batch_end += self.batch_size

            if self.current_batch_end > len(self.current_group_indices):
                self.current_batch_end = len(self.current_group_indices)
        return batch
