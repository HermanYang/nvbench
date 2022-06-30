import argparse
import sys
import data 
import torch
import os
from model import DLRM, DistributedDLRM
  
parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent)
parent = os.path.dirname(parent)
sys.path.append(parent)

import annotator

def dash_separated_ints(value):
    vals = value.split('-')
    for val in vals:
        try:
            int(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                '%s is not a valid dash separated list of ints' % value
            )
    return value


def run():
    parser = argparse.ArgumentParser(
        description='Deep Learning Recommendation Model (DLRM) Inference'
    )
    parser.add_argument('--category-feature-size', type=int, default=2)
    parser.add_argument(
        '--embedding', type=dash_separated_ints, default='4-3-2'
    )
    parser.add_argument('--mlp-bottom', type=dash_separated_ints, default='4-3-2')
    parser.add_argument('--mlp-top', type=dash_separated_ints, default='4-2-1')
    parser.add_argument('--interaction-itself', action='store_true', default=False)
    parser.add_argument('--activation-function', type=str, default='relu')
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--max-embedding-index', type=int, default=-1)
    parser.add_argument('--index-number-per-lookup', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--numpy-rand-seed', type=int, default=8867)
    parser.add_argument('--mode', type=str, default='throughput', choices=['latency', 'throughput'])

    args = parser.parse_args()
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))

    if local_world_size > 1:
        torch.distributed.init_process_group(backend='nccl')
    
    assert(torch.cuda.is_available())

    local_rank = int(os.environ.get('LOCAL_RANK', 0))  
    rank = int(os.environ.get('RANK', 0))  
    torch.cuda.manual_seed_all(args.numpy_rand_seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda', local_rank)
    batch_size = args.batch_size

    print('Using {} GPU(s)...'.format(local_rank))

    # parse model parameters
    bottom_mlp_input_output_size_list = list(map(int, args.mlp_bottom.split('-')))
    embedding_table_size_list = list(map(int, args.embedding.split('-')))
    numerical_feature_size = bottom_mlp_input_output_size_list[0]
    dataloader = data.make_random_data_loader(iterations=args.iterations, batch_size=batch_size, embedding_table_size_list=embedding_table_size_list, index_number_per_lookup=args.index_number_per_lookup, numerical_feature_size=numerical_feature_size, target_number=1, numerical_feature_dtype=torch.float16, category_feature_dtype=torch.int32)
    category_feature_size = args.category_feature_size
    interaction_input_size = len(embedding_table_size_list)+ 1
    bottom_mlp_output_size = bottom_mlp_input_output_size_list[len(bottom_mlp_input_output_size_list) - 1]
    bottom_output_size = (interaction_input_size * (interaction_input_size - 1)) // 2 + bottom_mlp_output_size
    arch_mlp_top_adjusted = str(bottom_output_size) + '-' + args.mlp_top
    top_mlp_input_size_list = list(map(int, arch_mlp_top_adjusted.split('-')))

    assert(args.max_embedding_index > args.index_number_per_lookup)

    # sanity check: feature sizes and mlp dimensions must match
    if numerical_feature_size != bottom_mlp_input_output_size_list[0]:
        sys.exit(
            'ERROR: arch-dense-feature-size '
            + str(numerical_feature_size)
            + ' does not match first dim of bottom mlp '
            + str(bottom_mlp_input_output_size_list[0])
        )

    if bottom_output_size != top_mlp_input_size_list[0]:
        sys.exit(
            'ERROR: # of feature interactions '
            + str(bottom_output_size)
            + ' does not match first dimension of top mlp '
            + str(top_mlp_input_size_list[0])
        )

    if local_world_size > 1:
        dlrm = DistributedDLRM(
            category_feature_size,
            embedding_table_size_list,
            bottom_mlp_input_output_size_list,
            top_mlp_input_size_list,
            device=device,
            dtype=torch.float16
        )
    else:
        dlrm = DLRM(
            category_feature_size,
            embedding_table_size_list,
            bottom_mlp_input_output_size_list,
            top_mlp_input_size_list,
            device=device,
            dtype=torch.float16
        )

    print(dlrm)

    # generate synthetic data in memory
    inputs=[]
    for input in dataloader:
        inputs.append(input)

    inference(rank, local_rank, dlrm, batch_size, inputs)

def inference(rank, local_rank, model, batch_size, inputs) -> list:
    outputs=[]
    input_count = len(inputs)
    for i in range(0, input_count):
        torch.distributed.barrier()
        annotator.inference_begin(rank=rank, local_rank=local_rank, index=i, batch_size=batch_size)
        numerical_feature_batch, embedding_index_batch_list, embedding_offset_batch_list = inputs[i]
        output = model(numerical_feature_batch, embedding_index_batch_list, embedding_offset_batch_list)
        if output is not None:
            annotator.output_begin(output)
            output = output.to('cpu', non_blocking=False)
            annotator.output_end()
            outputs.append(output)
        annotator.inference_end()
        torch.distributed.barrier()
    return outputs


def inference_optimized_for_throughput(model, inputs, device) -> list:
    outputs=[]
    input_count = len(inputs)
    for i in range(0, input_count):
        numerical_feature_batch, embedding_index_batch_list, embedding_offset_batch_list = inputs[i]
        # input tensors preload
        if i != input_count - 1:
            next_numerical_features, next_embedding_index_batch_list, next_embedding_offset_batch_list = inputs[i+1]
            next_numerical_features.to(device, non_blocking=True)
            for index_batch in next_embedding_index_batch_list:
                index_batch.to(device, non_blocking=True)
            for offset_batch in next_embedding_offset_batch_list:
                offset_batch.to(device, non_blocking=True)
        output = model(numerical_feature_batch, embedding_index_batch_list, embedding_offset_batch_list)
        if i != input_count - 1:
            output = output.to('cpu', non_blocking=True)
        else:
            output = output.to('cpu', non_blocking=False)
        outputs.append(output)
    return outputs


def distributed_inference_optimized_for_throughput(model, inputs, device) -> list:
    outputs=[]
    input_count = len(inputs)
    local_rank = 0
    for i in range(0, input_count):
        numerical_feature_batch, embedding_index_batch_list, embedding_offset_batch_list = inputs[i]
        # input tensors preload
        if i != input_count - 1:
            next_numerical_features, next_embedding_index_batch_list, next_embedding_offset_batch_list = inputs[i+1]
            if model.bottom_mlp:
                next_numerical_features.to(device, non_blocking=True)
            for i, index_batch in enumerate(next_embedding_index_batch_list):
                if i in model.embedding_bags_map[local_rank]:
                    index_batch.to(device, non_blocking=True)
            for i, offset_batch in enumerate(next_embedding_offset_batch_list):
                if i in model.embedding_bags_map[local_rank]:
                    offset_batch.to(device, non_blocking=True)
        output = model(numerical_feature_batch, embedding_index_batch_list, embedding_offset_batch_list)
        if i != input_count - 1:
            output = output.to('cpu', non_blocking=True)
        else:
            output = output.to('cpu', non_blocking=False)
        outputs.append(output)
    return outputs


if __name__ == '__main__':
    run()