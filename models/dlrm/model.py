import torch
import sys
import torch
import os

parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent)
parent = os.path.dirname(parent)
sys.path.append(parent)

import annotator

class DLRM(torch.nn.Module):
    def __init__(
        self,
        category_feature_size=None,
        embedding_table_size_list=None,
        bottom_mlp_input_output_size_list=None,
        top_mlp_input_size_list=None,
        device=None,
        dtype=torch.float32,
    ):
        super(DLRM, self).__init__()
        if (
            (category_feature_size is not None)
            and (embedding_table_size_list is not None)
            and (bottom_mlp_input_output_size_list is not None)
            and (top_mlp_input_size_list is not None)
            and (device is not None)
        ):
            self.device = device
            self.category_feature_size = category_feature_size 
            self.bottom_mlp = self.create_mlp(bottom_mlp_input_output_size_list, dtype=dtype)
            self.embeddings = self.create_emb(category_feature_size, embedding_table_size_list, dtype=dtype)
            self.top_mlp = self.create_mlp(top_mlp_input_size_list, dtype=dtype)

            # prepare the indices for the tril matrix
            interaction_input_size = 1 + len(embedding_table_size_list)
            self._tril_indices = torch.tensor([[i for i in range(interaction_input_size) for _ in range(i)], [j for i in range(interaction_input_size) for j in range(i)]], device=device)

    def create_mlp(self, linear_input_output_list, dtype):
        layers = torch.nn.ModuleList()
        for i in range(0, len(linear_input_output_list) - 1):
            input_feature_size = linear_input_output_list[i]
            output_feature_size = linear_input_output_list[i + 1]
            LinearLayer = torch.nn.Linear(input_feature_size, output_feature_size, bias=True, device=self.device, dtype=dtype)
            torch.nn.init.normal_(LinearLayer.weight, mean=0, std=0.1)
            torch.nn.init.normal_(LinearLayer.bias, mean=0, std=0.1)
            layers.append(LinearLayer)
            layers.append(torch.nn.ReLU())
        return torch.nn.Sequential(*layers)

    def create_emb(self, feature_size, embedding_table_size_list, dtype):
        embeddings = torch.nn.ModuleList()
        for table_size in  embedding_table_size_list:
            embedding = torch.nn.EmbeddingBag(table_size, feature_size, mode='sum', sparse=False, device=self.device, dtype=dtype)
            torch.nn.init.normal_(embedding.weight, mean=0, std=0.1)
            embeddings.append(embedding)
        return embeddings
    
    def apply_mlp(self, layers, x ):
        x = layers(x)
        return x

    def apply_emb(self, embeddings, embedding_index_batch_list, embedding_offset_batch_list):
        outputs = []
        for i, embedding in enumerate(embeddings):
            annotator.compute_begin('embedding', embedding)
            feature = embedding(
                embedding_index_batch_list[i], embedding_offset_batch_list[i]
            )
            annotator.compute_end()
            outputs.append(feature)
        return outputs 

    def interact_features(self, interaction_inputs):
        # interaction
        interaction_output = torch.bmm(interaction_inputs, torch.transpose(interaction_inputs, 1, 2))
        # flatten the interaction matrix, only keep the down-triangle part
        interaction_flat = interaction_output[:, self._tril_indices[0], self._tril_indices[1]]
        return interaction_flat 

    def forward(self, numerical_feature_batch, embedding_index_batch_list, embedding_offset_batch_list):
        batch_size = numerical_feature_batch.shape[0]

        # copy input
        if numerical_feature_batch.device != self.device:
            annotator.input_begin(numerical_feature_batch)
            numerical_feature_batch = numerical_feature_batch.to(self.device, non_blocking=True)
            annotator.input_end()

        for i in range(0, len(self.embeddings)):
            index_batch = embedding_index_batch_list[i]
            offset_batch = embedding_offset_batch_list[i]
            if index_batch.device != self.device:
                annotator.input_begin(index_batch)
                index_batch = index_batch.to(self.device, non_blocking=True)
                annotator.input_end()
                embedding_index_batch_list[i] = index_batch

            if embedding_offset_batch_list[i].device != self.device:
                annotator.input_begin(offset_batch)
                offset_batch = offset_batch.to(self.device, non_blocking=True)
                annotator.input_end()
                embedding_offset_batch_list[i] = offset_batch

        # bottom mlp
        annotator.compute_begin('bottom_mlp', self.bottom_mlp)
        bottom_mlp_output = self.apply_mlp(self.bottom_mlp, numerical_feature_batch)
        annotator.compute_end()

        embedding_bag_output_list = self.apply_emb(self.embeddings, embedding_index_batch_list, embedding_offset_batch_list)

        annotator.compute_begin('concatenation')
        interaction_input = torch.cat([bottom_mlp_output] + embedding_bag_output_list, dim=1).view((batch_size, -1, self.category_feature_size))
        annotator.compute_end()

        annotator.compute_begin('interaction')
        interaction_output = self.interact_features(interaction_input)
        annotator.compute_end()

        annotator.compute_begin('concatenation')
        bottom_output = torch.cat([bottom_mlp_output] + [interaction_output], dim=1)
        annotator.compute_end()

        annotator.compute_begin('top_mlp', self.top_mlp)
        output = self.apply_mlp(self.top_mlp, bottom_output)
        annotator.compute_end()

        return output 

class DistributedDLRM(DLRM):
    def __init__(
        self,
        category_feature_size=None,
        embedding_table_size_list=None,
        bottom_mlp_input_output_size_list=None,
        top_mlp_input_size_list=None,
        device=None,
        dtype=torch.float32,
    ):
        super(DistributedDLRM, self).__init__()
        if (
            (category_feature_size is not None)
            and (embedding_table_size_list is not None)
            and (bottom_mlp_input_output_size_list is not None)
            and (top_mlp_input_size_list is not None)
            and (device is not None)
        ):
            self.local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            assert(self.local_world_size > 1)
            self.device = device
            self.dtype = dtype
            self.category_feature_size = category_feature_size 
            self.bottom_mlp = None
            self.embeddings = None
            self.embedding_bags_map = {i: set() for i in range(0, self.local_world_size)}
            for i in range(0, len(embedding_table_size_list)):
                rank = (i % (self.local_world_size - 1)) + 1
                self.embedding_bags_map[rank].add(i)

            if self.local_rank == 0:
                self.bottom_mlp = self.create_mlp(bottom_mlp_input_output_size_list, dtype=dtype)
            else:
                embedding_table_size_list_local_rank = []
                for i, size in enumerate(embedding_table_size_list):
                    if i in self.embedding_bags_map[self.local_rank]:
                        embedding_table_size_list_local_rank.append(size)
                self.embeddings = self.create_emb(category_feature_size, embedding_table_size_list_local_rank, dtype=dtype)

            self.top_mlp = torch.nn.parallel.DistributedDataParallel(self.create_mlp(top_mlp_input_size_list, dtype=dtype))
            # prepare the indices for the tril matrix
            interaction_input_size = 1 + len(embedding_table_size_list)
            self._tril_indices = torch.tensor([[i for i in range(interaction_input_size) for _ in range(i)], [j for i in range(interaction_input_size) for j in range(i)]], device=device)

    def forward(self, numerical_feature_batch, embedding_index_batch_list, embedding_offset_batch_list):
        batch_size = numerical_feature_batch.shape[0]

        if batch_size >= self.local_world_size:
            batch_size_per_rank, remainder_batch_size = divmod(batch_size, self.local_world_size)
            assert remainder_batch_size == 0
        else:
            # batch_size can't smaller than 1, in this case, some of the worker won't perform to interaction and top mlp due to not enough data
            batch_size_per_rank = 1

        if self.local_rank == 0:
            # perform bottom mlp on local rank 0
            if numerical_feature_batch.device != self.device:
                annotator.input_begin(numerical_feature_batch)
                numerical_feature_batch = numerical_feature_batch.to(self.device)
                annotator.input_end()

            annotator.compute_begin('bottom_mlp', self.bottom_mlp)
            bottom_mlp_output = self.apply_mlp(self.bottom_mlp, numerical_feature_batch)
            annotator.compute_end()

            annotator.compute_begin('concatenation')
            scatter_tensors = list(bottom_mlp_output.split(batch_size_per_rank, dim = 0))
            gathered_tensors = [torch.empty((batch_size_per_rank, self.category_feature_size), device=self.device, dtype=self.dtype)]
            for i in range(1, self.local_world_size):
                gathered_tensors.append(torch.empty((batch_size_per_rank * len(self.embedding_bags_map[i]), self.category_feature_size), device=self.device, dtype=self.dtype))
            annotator.compute_end()

            # scatter empty tensor for small batch size
            if len(scatter_tensors) < self.local_world_size:
                for i in range(len(scatter_tensors), self.local_world_size):
                    scatter_tensors.append(torch.empty(0, device=self.device, dtype=self.dtype))

            annotator.communication_begin('all_to_all', recv_tensor_list=gathered_tensors, send_tensor_list=scatter_tensors)
            torch.distributed.all_to_all(gathered_tensors, scatter_tensors)
            annotator.communication_end()
        else:
            # perform embdedding bags on other ranks
            embedding_index_batch_list_local_rank= []
            embedding_offset_batch_list_local_rank= []
            for i in self.embedding_bags_map[self.local_rank]:
                index_batch = embedding_index_batch_list[i]
                if index_batch.device != self.device:
                    annotator.input_begin(index_batch)
                    index_batch = index_batch.to(self.device)
                    annotator.input_end()
                embedding_index_batch_list_local_rank.append(index_batch)
                offset_batch = embedding_offset_batch_list[i]
                if offset_batch.device != self.device:
                    annotator.input_begin(offset_batch)
                    offset_batch = offset_batch.to(self.device)
                    annotator.input_end()
                embedding_offset_batch_list_local_rank.append(offset_batch)
            embedding_bag_output_list = self.apply_emb(self.embeddings, embedding_index_batch_list_local_rank, embedding_offset_batch_list_local_rank)

            annotator.compute_begin('concatenation')
            embedding_bag_output = torch.cat(embedding_bag_output_list, dim=1)
            scatter_tensors = list(embedding_bag_output.split(batch_size_per_rank, dim = 0))
            if self.local_rank < batch_size:
                gathered_tensors = [torch.empty((batch_size_per_rank, self.category_feature_size), device=self.device, dtype=self.dtype)]
                for i in range(1, self.local_world_size):
                    gathered_tensors.append(torch.empty((batch_size_per_rank * len(self.embedding_bags_map[i]), self.category_feature_size), device=self.device, dtype=self.dtype))
            else:
                gathered_tensors = []
                for i in range(0, self.local_world_size):
                    gathered_tensors.append(torch.empty((0), device=self.device, dtype=self.dtype))
            annotator.compute_end()

            # scatter empty tensor for small batch size
            if len(scatter_tensors) < self.local_world_size:
                for i in range(len(scatter_tensors), self.local_world_size):
                    scatter_tensors.append(torch.empty(0, device=self.device, dtype=self.dtype))

            annotator.communication_begin('all_to_all', recv_tensor_list=gathered_tensors, send_tensor_list=scatter_tensors)
            torch.distributed.all_to_all(gathered_tensors, scatter_tensors)
            annotator.communication_end()

        if self.local_rank >= batch_size:
            return None

        annotator.compute_begin('concatenation')
        bottom_mlp_output=gathered_tensors[0]
        embedding_bag_output_list = []
        for i in range(1, self.local_world_size):
            embedding_bag_output_list.extend(list(gathered_tensors[i].chunk(len(self.embedding_bags_map[i]), dim=0)))
        interaction_input = torch.cat([bottom_mlp_output] + embedding_bag_output_list, dim=1).view((batch_size_per_rank, -1, self.category_feature_size))
        annotator.compute_end()

        annotator.compute_begin('interaction')
        interaction_output = self.interact_features(interaction_input)
        annotator.compute_end()

        annotator.compute_begin('concatenation')
        bottom_output = torch.cat([bottom_mlp_output] + [interaction_output], dim=1)
        annotator.compute_end()

        annotator.compute_begin('top_mlp', self.top_mlp)
        output = self.apply_mlp(self.top_mlp, bottom_output)
        annotator.compute_end()

        return output