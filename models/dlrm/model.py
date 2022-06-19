import torch
import torch.nn as nn
import numpy as np
import sys
import nvtx
import torch

def get_embedding_slice(n):
    size = my_size - 1
    k, m = divmod(n, size)
    return slice(
        my_rank * k + min(my_rank, m), (my_rank + 1) * k + min(my_rank + 1, m), 1
    )

def get_embedding_lengths(n):
    size = my_size - 1
    k, m = divmod(n, size)
    splits = [(k + 1) if i < m else k for i in range(size)]
    if my_rank == size:
        my_len = 0
    else:
        my_len = splits[my_rank]
    return (my_len, splits)

def get_my_slice(n):
    k, m = divmod(n, my_size)
    return slice(
        my_rank * k + min(my_rank, m), (my_rank + 1) * k + min(my_rank + 1, m), 1
    )

def get_split_lengths(n):
    k, m = divmod(n, my_size)
    if m == 0:
        splits = None
        my_len = k
    else:
        splits = [(k + 1) if i < m else k for i in range(my_size)]
        my_len = splits[my_rank]
    return (my_len, splits)


class DLRM(nn.Module):
    def create_mlp(self, ln, dtype):
        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]
            LL = nn.Linear(int(n), int(m), bias=True)
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(dtype)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(dtype)
            LL.weight.data = torch.tensor(W, requires_grad=False)
            LL.bias.data = torch.tensor(bt, requires_grad=False)
            layers.append(LL)
            layers.append(nn.ReLU())
        return torch.nn.Sequential(*layers)

    def create_emb(self, m, ln, dtype):
        emb_l = nn.ModuleList()
        v_W_l = []
        for i in range(0, ln.size):
            if i not in self.local_emb_indices:
                continue
            n = ln[i]
            EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
            EE.weight.data = torch.rand(n, m, dtype=dtype, requires_grad=True)
            emb_l.append(EE)
        return emb_l, v_W_l

    def __init__(
        self,
        m_spa=None,
        ln_emb=None,
        ln_bot=None,
        ln_top=None,
        arch_interaction_op=None,
        arch_interaction_itself=False,
        ndevices=-1,
    ):
        super(DLRM, self).__init__()

        if (
            (m_spa is not None)
            and (ln_emb is not None)
            and (ln_bot is not None)
            and (ln_top is not None)
            and (arch_interaction_op is not None)
        ):
            self.ndevices = ndevices
            self.output_d = 0
            self.parallel_model_batch_size = -1
            self.parallel_model_is_not_prepared = True
            self.arch_interaction_op = arch_interaction_op
            self.arch_interaction_itself = arch_interaction_itself
            self.sync_dense_params = sync_dense_params
            self.loss_threshold = loss_threshold
            self.loss_function=loss_function
            self.n_global_emb = len(ln_emb)

            if my_size > 1:
                n_emb = len(ln_emb)
                if n_emb < my_size:
                    sys.exit(
                        "only (%d) sparse features for (%d) devices, table partitions will fail"
                        % (n_emb, my_size)
                    )
                self.n_local_emb, self.n_emb_per_rank = get_embedding_lengths(
                    n_emb
                )
                self.local_emb_slice = get_embedding_slice(n_emb)
                self.local_emb_indices = list(range(n_emb))[self.local_emb_slice]

            self.m_spa = m_spa
            self.emb_l, w_list = self.create_emb(m_spa, ln_emb, torch.float16)
            self.v_W_l = w_list

            if my_rank == my_size - 1:
                self.bot_l = self.create_mlp(ln_bot, torch.float16)
            else:
                self.bot_l = None
            self.top_l = self.create_mlp(ln_top, torch.float16)

    def apply_mlp(self, x, layers):
        return layers(x)

    def apply_emb(self, lS_o, lS_i, emb_l, v_W_l):
        ly = []
        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = lS_o[k]
            E = emb_l[k]
            V = E(
                sparse_index_group_batch,
                sparse_offset_group_batch,
                per_sample_weights=None,
            )
            ly.append(V)
        return ly

    def interact_features(self, x, ly):
        if self.arch_interaction_op == "dot":
            (batch_size, d) = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            _, ni, nj = Z.shape
            offset = 1 if self.arch_interaction_itself else 0
            li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
            lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
            Zflat = Z[:, li, lj]
            R = torch.cat([x] + [Zflat], dim=1)
        elif self.arch_interaction_op == "cat":
            R = torch.cat([x] + ly, dim=1)
        else:
            sys.exit(
                "ERROR: --arch-interaction-op="
                + self.arch_interaction_op
                + " is not supported"
            )
        return R

    def forward(self, dense_x, lS_o, lS_i):
        return self.distributed_mix_parallel_forward(dense_x, lS_o, lS_i)

    @nvtx.annotate("inference")
    def distributed_mix_parallel_forward(self, dense_x, lS_o, lS_i):
        batch_size = dense_x.size()[0]
        # WARNING: # of ranks must be <= batch size in distributed_forward call
        if batch_size < my_size:
            sys.exit(
                "ERROR: batch_size (%d) must be larger than number of ranks (%d)"
                % (batch_size, my_size)
            )
        if batch_size % my_size != 0:
            sys.exit(
                "ERROR: batch_size %d can not split across %d ranks evenly"
                % (batch_size, my_size)
            )

        gathered_tensors = None
        device = f"cuda:{my_local_rank}" 
        sparse_vector_length = self.m_spa
        batch_size_per_rank = int(batch_size / my_size)
        # last rank perform bottom mlp, others perform embedding 
        if my_size > 1 and my_rank == my_size - 1:
            nvtx.push_range("numerical", category="input")
            dense_x = dense_x.to(device)
            nvtx.pop_range()
            nvtx.push_range("bottom mlp", category="compute")
            bottom_mlp_output = self.apply_mlp(dense_x, self.bot_l)
            nvtx.pop_range()
            scatter_tensors = list(bottom_mlp_output.split(batch_size_per_rank ,dim = 0))
            if gathered_tensors is None:
                gathered_tensors = [torch.empty((batch_size_per_rank * n, sparse_vector_length), device=device, dtype=torch.float16) for n in self.n_emb_per_rank]
                gathered_tensors.append(torch.empty((batch_size_per_rank, sparse_vector_length), device=device, dtype=torch.float16))
            nvtx.push_range("all to all", category="communication")
            torch.distributed.all_to_all(gathered_tensors, scatter_tensors)
            nvtx.pop_range()
        elif my_size > 1:
            nvtx.push_range("category", category="input")
            lS_o = [o.to(device) for o in lS_o[self.local_emb_slice]]
            lS_i = [i.to(device) for i in lS_i[self.local_emb_slice]]
            nvtx.pop_range()
            if (len(self.emb_l) != len(lS_o)) or (len(self.emb_l) != len(lS_i)):
                sys.exit(
                    "ERROR: corrupted model input detected in distributed_forward call"
                )
            nvtx.push_range("embedding", category="compute")
            embedding_output = self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l)
            nvtx.pop_range()
            embedding_output = torch.cat(embedding_output, dim=1)
            scatter_tensors = list(embedding_output.split(batch_size_per_rank, dim = 0))
            if gathered_tensors is None:
                gathered_tensors = [torch.empty((batch_size_per_rank * n, sparse_vector_length), device=device, dtype=torch.float16) for n in self.n_emb_per_rank]
                gathered_tensors.append(torch.empty((batch_size_per_rank, sparse_vector_length), device=device, dtype=torch.float16))
            nvtx.push_range("all to all", category="communication")
            torch.distributed.all_to_all(gathered_tensors, scatter_tensors)
            nvtx.pop_range()

        ly = []
        if my_size > 1:
            for i in range(len(self.n_emb_per_rank)):
                ly.extend(list(gathered_tensors[i].chunk(self.n_emb_per_rank[i], dim=0)))
            x =  gathered_tensors[my_size - 1]
        else:
            nvtx.push_range("numerical")
            dense_x = dense_x.to(device)
            nvtx.pop_range()
            nvtx.push_range("category")
            lS_o = [o.to(device) for o in lS_o]
            lS_i = [i.to(device) for i in lS_i]
            nvtx.pop_range()
            nvtx.push_range("bottom_mlp")
            x = self.apply_mlp(dense_x, self.bot_l)
            nvtx.pop_range()
            nvtx.push_range("embedding")
            ly = list(self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l))
            nvtx.pop_range()

        if self.n_global_emb != len(ly):
            sys.exit("ERROR: corrupted intermediate result in distributed_forward call")

        nvtx.push_range("interaction")
        z = self.interact_features(x, ly)
        nvtx.pop_range()

        nvtx.push_range("top_mlp")
        p = self.apply_mlp(z, self.top_l)
        nvtx.pop_range()

        return p

    @nvtx.annotate("inference")
    def sequential_forward(self, dense_x, lS_o, lS_i):
        nvtx.push_range("bottom_mlp")
        x = self.apply_mlp(dense_x, self.bot_l)
        nvtx.pop_range()

        nvtx.push_range("embedding_bag")
        ly = self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l)
        nvtx.pop_range()

        nvtx.push_range("interaction")
        z = self.interact_features(x, ly)
        nvtx.pop_range()

        nvtx.push_range("top_mlp")
        p = self.apply_mlp(z, self.top_l)
        nvtx.pop_range()

        return p