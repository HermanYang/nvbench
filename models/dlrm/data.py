from unicodedata import category
import torch
import numpy
from torch.utils.data import Dataset

def make_random_data_loader(iterations, batch_size, embedding_table_size_list, index_number_per_lookup, numerical_feature_size, target_number, numerical_feature_dtype, category_feature_dtype):
    dataset = RandomDataset(
        numerical_feature_size=numerical_feature_size,
        embedding_table_size_list=embedding_table_size_list,
        index_number_per_lookup=index_number_per_lookup,
        iterations=iterations,
        batch_size=batch_size,
        target_number=target_number,
        numerical_feature_dtype=numerical_feature_dtype,
        category_feature_dtype=category_feature_dtype
    )
    collate_wrapper_random = collate_wrapper_random_offset
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=len(embedding_table_size_list) * 2,
        collate_fn=collate_wrapper_random,
        pin_memory=True,
    )
    return data_loader

class RandomDataset(Dataset):
    def __init__(
            self,
            iterations,
            batch_size,
            numerical_feature_size,
            index_number_per_lookup,
            embedding_table_size_list,
            target_number=1,
            numerical_feature_dtype=torch.float32,
            category_feature_dtype=numpy.int32,
    ):
        self.numerical_feature_size = numerical_feature_size
        self.embedding_table_size_list = embedding_table_size_list
        self.index_number_per_lookup = index_number_per_lookup
        self.interations = iterations
        self.batch_size = batch_size
        self.target_number = target_number
        self.data_size = iterations * batch_size
        self.numerical_feature_dtype = numerical_feature_dtype
        self.category_feature_dtype = category_feature_dtype

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [
                self[idx] for idx in range(
                    index.start or 0, index.stop or len(self), index.step or 1
                )
            ]

        # generate input data
        (numerical_feature_batch, embedding_index_batch_list, embedding_offset_batch_list) = generate_input_data(self.numerical_feature_size, self.embedding_table_size_list, self.batch_size, self.index_number_per_lookup, self.numerical_feature_dtype, self.category_feature_dtype)

        return (numerical_feature_batch, embedding_index_batch_list, embedding_offset_batch_list)

    def __len__(self):
        return self.interations


def collate_wrapper_random_offset(list_of_tuples):
    (numerical_feature_batch, embedding_index_batch_list, embedding_offset_batch_list) = list_of_tuples[0]
    return (numerical_feature_batch, embedding_index_batch_list, embedding_offset_batch_list)


def generate_input_data(
    numerical_feature_size,
    embedding_table_size_list,
    batch_size,
    index_number_per_lookup,
    numerical_feature_dtype,
    category_feature_dtype
):
    # dense feature
    numerical_feature_batch = torch.tensor(numpy.random.rand(batch_size, numerical_feature_size), dtype=numerical_feature_dtype, device='cpu')

    # sparse feature (sparse indices)
    embedding_bag_offset_batch_list = []
    embedding_bag_index_batch_list = []

    # for each embedding generate a list of n lookups,
    # where each lookup is composed of multiple sparse indices
    for size in embedding_table_size_list:
        embedding_bag_offset_batch = []
        embedding_bag_index_batch = []
        offset = 0
        for _ in range(batch_size):
            random_lookup_indices = numpy.random.random(index_number_per_lookup * len(embedding_table_size_list))
            indices_per_lookup = numpy.unique(numpy.round(random_lookup_indices * (size - 1)))[:index_number_per_lookup]
            assert(index_number_per_lookup == indices_per_lookup.size)
            embedding_bag_offset_batch += [offset]
            embedding_bag_index_batch += indices_per_lookup.tolist()
            offset += index_number_per_lookup

        embedding_bag_offset_batch_list.append(torch.tensor(embedding_bag_offset_batch, device='cpu', dtype=category_feature_dtype))
        embedding_bag_index_batch_list.append(torch.tensor(embedding_bag_index_batch, device='cpu', dtype=category_feature_dtype))

    return (numerical_feature_batch, embedding_bag_index_batch_list, embedding_bag_offset_batch_list)