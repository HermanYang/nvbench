import argparse
import sqlite3
import pathlib
import pandas 
import matplotlib.pyplot as plt
import numpy
import json

parser = argparse.ArgumentParser()

parser.add_argument('--input', type=str, required=True, help='Path to nvbench results')
parser.add_argument('--output', type=str, required=True, help='Path analysis results')

def analyze(nsysdb_path, power_draw_path):
    create_stats_views_sql = pathlib.Path('analyzer/create_stats_views.sql').read_text()
    con = sqlite3.connect(nsysdb_path)
    cur = con.cursor()

    # Get from https://docs.nvidia.com/nsight-systems/UserGuide/index.html#exporter-sqlite-event-values
    nvtx_event_map = {
        33: 'NvtxCategory',
        34: 'NvtxMark',
        39: 'NvtxThread',
        59: 'NvtxPushPopRange',
        60: 'NvtxStartEndRange',
        75: 'NvtxDomainCreate',
        76: 'NvtxDomainDestroy'
    }

    query_string_map = ''' 
        SELECT id, value
        FROM StringIds
    '''

    con.create_function('nvtx_event_map', 1, lambda x: nvtx_event_map[x])
    string_map = dict(con.execute(query_string_map).fetchall())
    con.create_function('string_map', 1, lambda x: string_map[x])

    dtype_bytes_map = {'float32':4, 'float16':2, 'int32':4}
    def tensor_to_bytes(tensor_json):
        tensor = json.loads(tensor_json)
        if len(tensor) == 0:
            return 0
        shape = tensor[0]
        dtype = tensor[1]
        tensor_bytes = 1
        for dim in shape:
            tensor_bytes = tensor_bytes * dim
        return tensor_bytes * dtype_bytes_map[dtype]
    con.create_function('tensor_to_bytes', 1, tensor_to_bytes)

    def tensors_to_bytes(tensor_list_json):
        tensors = json.loads(tensor_list_json)
        if len(tensors) == 0:
            return 0
        total_tensor_bytes = 0
        for tensor in tensors:
            shape = tensor[0]
            dtype = tensor[1]
            tensor_bytes = 1
            for dim in shape:
                tensor_bytes = tensor_bytes * dim
            total_tensor_bytes += tensor_bytes * dtype_bytes_map[dtype]
        return total_tensor_bytes
    con.create_function('tensors_to_bytes', 1, tensors_to_bytes)

    worker_power_draw = []
    power_draw_df = pandas.DataFrame(pandas.read_csv(power_draw_path))
    worker_power_draw = power_draw_df.max().to_list()

    con.executescript(create_stats_views_sql)

    # Fool-proofing
    input_items = len(pandas.read_sql_query('SELECT * FROM step_input_range', con).index)
    assert len(pandas.read_sql_query('SELECT * FROM step_input_runtime', con).index) == input_items
    assert len(pandas.read_sql_query('SELECT * FROM step_input_kernel', con).index) == input_items

    output_items = len(pandas.read_sql_query('SELECT * FROM step_output_range', con).index)
    assert len(pandas.read_sql_query('SELECT * FROM step_output_runtime', con).index) == output_items
    assert len(pandas.read_sql_query('SELECT * FROM step_output_kernel', con).index) == output_items

    communication_items = len(pandas.read_sql_query('SELECT * FROM step_communication_range', con).index)
    assert len(pandas.read_sql_query('SELECT * FROM step_communication_runtime', con).index) == communication_items
    assert len(pandas.read_sql_query('SELECT * FROM step_communication_kernel', con).index) == communication_items
    assert len(pandas.read_sql_query('SELECT * FROM step_communication_wait_kernel', con).index) == communication_items
    assert len(pandas.read_sql_query('SELECT * FROM step_communication_transfer_kernel', con).index) == communication_items

    input_stats_df = pandas.read_sql_query('SELECT * FROM step_input_stats', con)
    output_stats_df = pandas.read_sql_query('SELECT * FROM step_output_stats', con)
    compute_stats_df = pandas.read_sql_query('SELECT * FROM step_compute_stats', con)
    comunication_wait_stats_df = pandas.read_sql_query('SELECT * FROM step_communication_wait_stats', con)
    comunication_transfer_stats_df = pandas.read_sql_query('SELECT * FROM step_communication_transfer_stats', con)
    inference_stats_df = pandas.read_sql_query('SELECT * FROM step_inference_stats', con)

    cur.close()

    # merge all stats
    worker_stats_df = inference_stats_df.merge(
        input_stats_df.rename(columns={'host_duration':'input_host_duration', 'device_duration':'input_device_duration', 'host_bytes':'input_host_bytes', 'device_bytes':'input_device_bytes', 'throughput':'input_throughput'}), 
        how='left', on=['rank', 'local_rank', 'globalTid','step_id'])
    worker_stats_df = worker_stats_df.merge(comunication_wait_stats_df.rename(columns={'host_duration':'communication_host_duration', 'device_duration':'communication_wait_device_duration', 'send_bytes':'communication_send_bytes', 'recv_bytes':'communication_recv_bytes'}), how='left', on=['rank', 'local_rank', 'globalTid', 'step_id'])
    worker_stats_df = worker_stats_df.merge(comunication_transfer_stats_df.filter(['rank', 'local_rank', 'globalTid','step_id', 'device_duration']).rename(columns={'device_duration':'communication_transfer_device_duration'}), how='left', on=['rank', 'local_rank', 'globalTid', 'step_id'])
    bottom_mlp_stats_df = compute_stats_df[compute_stats_df['event_value'] == 'bottom_mlp'].filter(items=['step_id', 'rank', 'local_rank', 'globalTid', 'weights', 'host_duration', 'device_duration'])
    embedding_stats_df = compute_stats_df[compute_stats_df['event_value'] == 'embedding'].filter(items=['step_id', 'rank', 'local_rank', 'globalTid', 'weights', 'host_duration', 'device_duration'])
    interaction_stats_df = compute_stats_df[compute_stats_df['event_value'] == 'interaction'].filter(items=['step_id', 'rank', 'local_rank', 'globalTid', 'weights', 'host_duration', 'device_duration'])
    top_mlp_stats_df = compute_stats_df[compute_stats_df['event_value'] == 'top_mlp'].filter(items=['step_id', 'rank', 'local_rank', 'globalTid', 'weights', 'host_duration', 'device_duration'])

    worker_stats_df = worker_stats_df.merge(bottom_mlp_stats_df.rename(columns={'weights':'bottom_mlp_weights', 'host_duration':'bottom_mlp_duration', 'device_duration':'bottom_mlp_device_duration'}), how='left',  on=['rank', 'local_rank','globalTid','step_id'])
    worker_stats_df = worker_stats_df.merge(embedding_stats_df.rename(columns={'weights':'embedding_weights', 'host_duration':'embedding_duration', 'device_duration':'embedding_device_duration'}), how='left', on=['rank', 'local_rank','globalTid','step_id'])
    worker_stats_df = worker_stats_df.merge(interaction_stats_df.rename(columns={'weights':'interaction_weights', 'host_duration':'interaction_host_duration', 'device_duration':'interaction_device_duration'} ), how='left', on=['rank', 'local_rank','globalTid','step_id'])
    worker_stats_df = worker_stats_df.merge(top_mlp_stats_df.rename(columns={'weights':'top_mlp_weights', 'host_duration':'top_mlp_host_duration', 'device_duration':'top_mlp_device_duration'}), how='left', suffixes=['', '_top_mlp'], on=['rank', 'local_rank','globalTid','step_id'])
    worker_stats_df = worker_stats_df.merge(output_stats_df.rename(columns={'host_duration':'output_host_duration', 'device_duration':'output_device_duration', 'host_bytes':'output_host_bytes', 'device_bytes':'output_device_bytes', 'throughput':'output_throughput'}), how='left', on=['rank', 'local_rank','globalTid','step_id'])
    worker_stats_df = worker_stats_df.fillna(0)

    # estimate device_duration, only consider input + compute + communication + output time on device
    worker_stats_df = worker_stats_df.assign(device_duration = lambda stats: stats['input_device_duration'] + stats['bottom_mlp_device_duration'] + stats['embedding_device_duration'] + stats['communication_transfer_device_duration'] + stats['interaction_device_duration'] + stats['top_mlp_device_duration'] + stats['output_device_duration'])
    # estimate device_duration_without_io, only consider compute + communication on device
    worker_stats_df = worker_stats_df.assign(device_duration_without_io = lambda stats: stats['bottom_mlp_device_duration'] + stats['embedding_device_duration'] + stats['communication_transfer_device_duration'] + stats['interaction_device_duration'] + stats['top_mlp_device_duration'])

    # server scenatio layer ratio
    ratio_to_percentage=100
    worker_stats_df = worker_stats_df.assign(server_input_percentage = lambda stats: (stats['input_device_duration'] / stats['device_duration']) * ratio_to_percentage)
    worker_stats_df = worker_stats_df.assign(server_output_percentage = lambda stats: (stats['output_device_duration'] / stats['device_duration']) * ratio_to_percentage)
    worker_stats_df = worker_stats_df.assign(server_bottom_mlp_percentage = lambda stats: (stats['bottom_mlp_device_duration'] / stats['device_duration']) * ratio_to_percentage)
    worker_stats_df = worker_stats_df.assign(server_embedding_percentage = lambda stats: (stats['embedding_device_duration'] / stats['device_duration']) * ratio_to_percentage)
    worker_stats_df = worker_stats_df.assign(server_communication_percentage = lambda stats: (stats['communication_transfer_device_duration'] / stats['device_duration']) * ratio_to_percentage)
    worker_stats_df = worker_stats_df.assign(server_interaction_percentage = lambda stats: (stats['interaction_device_duration'] / stats['device_duration']) * ratio_to_percentage)
    worker_stats_df = worker_stats_df.assign(server_top_mlp_percentage = lambda stats: (stats['top_mlp_device_duration'] / stats['device_duration']) * ratio_to_percentage)

    # offline scenatio layer ratio
    worker_stats_df = worker_stats_df.assign(offline_bottom_mlp_percentage = lambda stats: (stats['bottom_mlp_device_duration'] / stats['device_duration_without_io']) * ratio_to_percentage)
    worker_stats_df = worker_stats_df.assign(offline_embedding_percentage = lambda stats: (stats['embedding_device_duration'] / stats['device_duration_without_io']) * ratio_to_percentage)
    worker_stats_df = worker_stats_df.assign(offline_communication_percentage = lambda stats: (stats['communication_transfer_device_duration'] / stats['device_duration_without_io']) * ratio_to_percentage)
    worker_stats_df = worker_stats_df.assign(offline_interaction_percentage = lambda stats: (stats['interaction_device_duration'] / stats['device_duration_without_io']) * ratio_to_percentage)
    worker_stats_df = worker_stats_df.assign(offline_top_mlp_percentage = lambda stats: (stats['top_mlp_device_duration'] / stats['device_duration_without_io']) * ratio_to_percentage)

    # create system stats from all worker stats
    system_stats_df = worker_stats_df
    system_stats_df = system_stats_df.groupby(['step_id']).agg(
        batch_size=('batch_size', 'max'),
        worker_number=('rank', 'count'),
        local_worker_number=('local_rank', 'count'),
        host_start =('host_start', 'min'), 
        host_end =('host_end', 'max'), 
        input_device_duration_sum = ('input_device_duration', 'sum'),
        input_bytes = ('input_device_bytes', 'sum'),
        send_bytes = ('communication_send_bytes', 'sum'),
        recv_bytes = ('communication_recv_bytes', 'sum'),
        communication_transfer_device_duration_sum = ('communication_transfer_device_duration', 'sum'),
        bottom_mlp_weights = ('bottom_mlp_weights', 'sum'),
        bottom_mlp_device_duration_sum = ('bottom_mlp_device_duration', 'sum'),
        embedding_weights = ('embedding_weights', 'sum'),
        embedding_device_duration_sum = ('embedding_device_duration', 'sum'),
        interaction_weights = ('interaction_weights', 'sum'),
        interaction_device_duration_sum = ('interaction_device_duration', 'sum'),
        top_mlp_weights = ('top_mlp_weights', 'sum'),
        top_mlp_device_duration_sum = ('top_mlp_device_duration', 'sum'),
        output_device_duration_sum = ('output_device_duration', 'sum'),
        output_bytes = ('output_device_bytes', 'sum'),
        device_duration=('device_duration', 'max'), 
        device_duration_sum =('device_duration', 'sum'), 
        device_duration_without_io=('device_duration_without_io', 'max'), 
        device_duration_without_io_sum=('device_duration_without_io', 'sum'))
    
    # system level layer percentage
    system_stats_df = system_stats_df.assign(server_input_percentage = lambda stats: (stats['input_device_duration_sum'] / stats['device_duration_sum']) * ratio_to_percentage)
    system_stats_df = system_stats_df.assign(server_output_percentage = lambda stats: (stats['output_device_duration_sum'] / stats['device_duration_sum']) * ratio_to_percentage)
    system_stats_df = system_stats_df.assign(server_bottom_mlp_percentage = lambda stats: (stats['bottom_mlp_device_duration_sum'] / stats['device_duration_sum']) * ratio_to_percentage)
    system_stats_df = system_stats_df.assign(server_embedding_percentage = lambda stats: (stats['embedding_device_duration_sum'] / stats['device_duration_sum']) * ratio_to_percentage)
    system_stats_df = system_stats_df.assign(server_communication_percentage = lambda stats: (stats['communication_transfer_device_duration_sum'] / stats['device_duration_sum']) * ratio_to_percentage)
    system_stats_df = system_stats_df.assign(server_interaction_percentage = lambda stats: (stats['interaction_device_duration_sum'] / stats['device_duration_sum']) * ratio_to_percentage)
    system_stats_df = system_stats_df.assign(server_top_mlp_percentage = lambda stats: (stats['top_mlp_device_duration_sum'] / stats['device_duration_sum']) * ratio_to_percentage)

    system_stats_df = system_stats_df.assign(offline_input_percentage = lambda stats: (stats['input_device_duration_sum'] / stats['device_duration_without_io_sum']) * ratio_to_percentage)
    system_stats_df = system_stats_df.assign(offline_output_percentage = lambda stats: (stats['output_device_duration_sum'] / stats['device_duration_without_io_sum']) * ratio_to_percentage)
    system_stats_df = system_stats_df.assign(offline_bottom_mlp_percentage = lambda stats: (stats['bottom_mlp_device_duration_sum'] / stats['device_duration_without_io_sum']) * ratio_to_percentage)
    system_stats_df = system_stats_df.assign(offline_embedding_percentage = lambda stats: (stats['embedding_device_duration_sum'] / stats['device_duration_without_io_sum']) * ratio_to_percentage)
    system_stats_df = system_stats_df.assign(offline_communication_percentage = lambda stats: (stats['communication_transfer_device_duration_sum'] / stats['device_duration_without_io_sum']) * ratio_to_percentage)
    system_stats_df = system_stats_df.assign(offline_interaction_percentage = lambda stats: (stats['interaction_device_duration_sum'] / stats['device_duration_without_io_sum']) * ratio_to_percentage)
    system_stats_df = system_stats_df.assign(offline_top_mlp_percentage = lambda stats: (stats['top_mlp_device_duration_sum'] / stats['device_duration_without_io_sum']) * ratio_to_percentage)

    system_stats_df = system_stats_df.assign(host_duration = lambda stats: stats['host_end'] - stats['host_start'])
    system_stats_df = system_stats_df.drop(columns=['host_start', 'host_end'])

    nanosecond_to_second = 1e-9
    system_stats_df = system_stats_df.assign(input_throughput = lambda stats: stats['input_bytes'] / (stats['input_device_duration_sum'] * nanosecond_to_second))
    system_stats_df = system_stats_df.assign(output_throughput = lambda stats: stats['output_bytes'] / (stats['output_device_duration_sum'] * nanosecond_to_second))
    system_stats_df = system_stats_df.assign(communication_throughput = lambda stats: (stats['send_bytes'] + stats['recv_bytes']) / (stats['communication_transfer_device_duration_sum'] * nanosecond_to_second))
    system_stats_df = system_stats_df.assign(ips = lambda stats: stats['batch_size'] / (stats['host_duration'] * nanosecond_to_second))
    system_stats_df = system_stats_df.assign(ideal_ips = lambda stats: stats['batch_size'] / (stats['device_duration'] * nanosecond_to_second))
    system_stats_df = system_stats_df.assign(ideal_ips_without_io = lambda stats: stats['batch_size'] / (stats['device_duration_without_io'] * nanosecond_to_second))

    # power draw estimation with real ips
    system_stats_df = system_stats_df.assign(ipw = lambda stats: stats['ips'] / sum(worker_power_draw))

    return worker_stats_df, system_stats_df


def main():
    args = parser.parse_args()
    input = pathlib.Path(args.input)
    output = pathlib.Path(args.output)

    if not input.exists():
        raise ValueError('Input input does not exist: {}'.format(input))

    if not output.exists():
        raise ValueError('Output input does not exist: {}'.format(input))

    system_stats_df = pandas.DataFrame()
    worker_stats_df = pandas.DataFrame()

    for file in input.glob('**/*.sqlite'):
        nsysdb_path = f'{input}/{file.stem}.sqlite'
        power_draw_path = f'{input}/{file.stem}_power_draw.csv'

        if not pathlib.Path(nsysdb_path).exists():
            raise ValueError('Path does not exist: {}'.format(nsysdb_path))

        if not pathlib.Path(power_draw_path).exists():
            raise ValueError('Path does not exist: {}'.format(power_draw_path))

        worker_stats_per_batch_df, system_stats_per_batch_df = analyze(nsysdb_path, power_draw_path)
        system_stats_df = pandas.concat([system_stats_df, system_stats_per_batch_df])
        worker_stats_df = pandas.concat([worker_stats_df, worker_stats_per_batch_df])

    worker_stats_df.to_csv(f'{output}/worker_stats.csv')
    system_stats_df.to_csv(f'{output}/system_stats.csv')


    offline_layer_percentage_df = worker_stats_df[worker_stats_df['step_id'] == worker_stats_df['step_id'].max()].filter(items=['batch_size', 'local_rank', 'offline_bottom_mlp_percentage', 'offline_embedding_percentage', 'offline_communication_percentage', 'offline_interaction_percentage', 'offline_top_mlp_percentage'])
    offline_layer_percentage_df.to_csv(f'{output}/offline_layer_percentage.csv', header=['Batch Size', 'Local Rank', 'Bottom MLP(%)', 'Emdedding(%)', 'Communication(%)', 'Interaction(%)', 'Top MLP(%)'], index=False)

    server_layer_percentage_df = worker_stats_df[worker_stats_df['step_id'] == worker_stats_df['step_id'].max()].filter(items=['batch_size', 'local_rank', 'server_input_percentage', 'server_bottom_mlp_percentage', 'server_embedding_percentage', 'server_communication_percentage', 'server_interaction_percentage', 'server_top_mlp_percentage', 'server_output_percentage'])
    server_layer_percentage_df.to_csv(f'{output}/server_layer_percentage.csv', header=['Batch Size', 'Local Rank', 'Input(%)', 'Bottom MLP(%)', 'Embedding(%)', 'Communication(%)', 'Interaction(%)', 'Top MLP(%)', 'Output(%)'],index=False)

    layer_system_latency_df = worker_stats_df[worker_stats_df['step_id'] == worker_stats_df['step_id'].max()].filter(items=['batch_size', 'local_rank', 'input_device_duration', 'bottom_mlp_device_duration', 'embedding_device_duration', 'communication_transfer_device_duration', 'interaction_device_duration', 'top_mlp_device_duration', 'output_device_duration'])

    # System Level
    # IPS
    system_ips_df = system_stats_df[system_stats_df.index == system_stats_df.index.max()].filter(items=['batch_size', 'ips', 'ideal_ips', 'ideal_ips_without_io']).sort_values('batch_size').reset_index(drop=True)
    # IPW
    system_ipw_df = system_stats_df[system_stats_df.index == system_stats_df.index.max()].filter(items=['batch_size', 'ipw']).sort_values('batch_size').reset_index(drop=True)
    # E2E latency
    system_latency_df = system_stats_df[system_stats_df.index == system_stats_df.index.max()].filter(items=['batch_size', 'host_duration', 'device_duration', 'device_duration_without_io']).sort_values('batch_size').reset_index(drop=True)
    # IO bytes and Throughput
    system_io_df = system_stats_df[system_stats_df.index == system_stats_df.index.max()].filter(items=['batch_size', 'input_bytes', 'output_bytes', 'input_throughput', 'output_throughput']).sort_values('batch_size').reset_index(drop=True)
    # Communication bytes and Throughput
    system_communication_df = system_stats_df[system_stats_df.index == system_stats_df.index.max()].filter(items=['batch_size', 'send_bytes', 'recv_bytes', 'communication_throughput']).sort_values('batch_size').reset_index(drop=True)
    # Layer Weights 
    system_layer_weight_df = system_stats_df[system_stats_df.index == system_stats_df.index.max()].filter(items=['batch_size', 'bottom_mlp_weights', 'embedding_weights', 'interaction_weights', 'top_mlp_weights']).sort_values('batch_size').reset_index(drop=True)
    # Layer Percentage
    system_layer_percentage_df = system_stats_df[system_stats_df.index == system_stats_df.index.max()].filter(items=['batch_size', 'server_input_percentage','server_bottom_mlp_percentage', 'server_embedding_percentage', 'server_communication_percentage', 'server_interaction_percentage', 'server_top_mlp_percentage', 'server_output_percentage']).sort_values('batch_size').reset_index(drop=True)
    # Layer Percentage without IO
    system_layer_percentage_without_io_df = system_stats_df[system_stats_df.index == system_stats_df.index.max()].filter(items=['batch_size', 'offline_bottom_mlp_percentage', 'offline_embedding_percentage', 'offline_communication_percentage', 'offline_interaction_percentage', 'offline_top_mlp_percentage']).sort_values('batch_size').reset_index(drop=True)
    # Layer Latency 
    system_layer_latency_df = system_stats_df[system_stats_df.index == system_stats_df.index.max()].filter(items=['batch_size', 'input_device_duration_sum', 'bottom_mlp_device_duration_sum', 'embedding_device_duration_sum', 'interaction_device_duration_sum', 'top_mlp_device_duration_sum', 'output_device_duration_sum']).sort_values('batch_size').reset_index(drop=True)

    # Plotting
    def line(df, x, y, title, xlabel, ylabel, legend, path, logy = False):
        kind = 'line'
        plt.clf()
        df=df.copy(deep=True)
        xtick_labels = df[x].to_numpy().astype(str)
        df[x] = df[x].transform(numpy.log2)
        axs = df.plot(x=x, y=y, kind=kind, grid=True, marker = 'o', title=title, xlabel=xlabel, ylabel=ylabel, logy = logy)
        axs.set_xticks(df[x].to_numpy())
        axs.set_xticklabels(labels = xtick_labels)
        plt.legend(legend)
        plt.tight_layout()
        plt.savefig(path)

    def bar(df, x, y, title, xlabel, ylabel, legend, path, logy = False, stacked = False):
        kind = 'bar'
        plt.clf()
        df=df.copy(deep=True)
        xtick_labels = df[x].to_numpy().astype(str)
        df[x] = df[x].transform(numpy.log2)
        axs = df.plot(x=x, y=y, kind=kind, grid=True, title=title, xlabel=xlabel, ylabel=ylabel, logy = logy, stacked = stacked)
        axs.set_xticks(df[x].to_numpy())
        axs.set_xticklabels(labels = xtick_labels)
        plt.legend(legend)
        plt.tight_layout()
        plt.savefig(path)

    #IPS
    header =['Batch Size', 'IPS', 'Ideal IPS', 'Ideal IPS without IO']
    system_ips_df.to_csv(f'{output}/system_ips.csv', header = header, index=False)
    line(df=system_ips_df, 
        x='batch_size',
        y=['ips', 'ideal_ips', 'ideal_ips_without_io'],
        title='System Throughput', 
        xlabel=header[0], 
        ylabel='Inference Per Second', 
        legend=header[1:], 
        path=f'{output}/system_ips.png')

    # IPW
    header=['Batch Size', 'IPW']
    system_ipw_df.to_csv(f'{output}/system_ipw.csv', header = header, index=False)
    line(df=system_ipw_df,
        x='batch_size', 
        y=['ipw'], 
        title='System Energey Efficency', 
        xlabel=header[0], 
        ylabel='Inference Per Watt', 
        legend=header[1:], 
        path=f'{output}/system_ipw.png')

    # E2E Latency
    header=['Batch Size', 'Latency(ns)', 'Device Latency(ns)', 'Device Latency Without IO(ns)']
    system_latency_df.to_csv(f'{output}/system_e2e_latency.csv', header = header, index=False)
    line(df=system_latency_df, 
        x='batch_size', 
        y=['host_duration', 'device_duration', 'device_duration_without_io'], 
        title='System E2E Latency', 
        xlabel=header[0], 
        ylabel='Latency(ns)', 
        legend=header[1:], 
        path=f'{output}/system_e2e_latency.png')

    # IO bytes
    header=['Batch Size', 'Input(bytes)', 'Output(bytes)', 'Input(bytes/s)', 'Output(bytes/s)']
    system_io_df.to_csv(f'{output}/system_io.csv', header = header, index=False)
    line(df=system_io_df, 
        x='batch_size', 
        y=['input_bytes', 'output_bytes'], 
        title='System IO Bytes',
        xlabel=header[0], 
        ylabel='Bytes', 
        legend=header[1:3], 
        path=f'{output}/system_io_bytes.png')

    # IO Throughput 
    line(df=system_io_df, 
        x='batch_size', 
        y=['input_throughput', 'output_throughput'], 
        title='System IO Throughput', 
        xlabel=header[0], 
        ylabel='Bytes Per Second', 
        legend=header[3:], 
        path=f'{output}/system_io_throughput.png')

    # Communication bytes
    bar(df=system_communication_df, 
        x='batch_size', 
        y=['send_bytes', 'recv_bytes'], 
        title='System Communication Bytes', 
        xlabel='Batch Size', 
        ylabel='Bytes', 
        legend=['Send', 'Receive'], 
        path=f'{output}/system_communication_bytes.png')

    # Communication Throughtput
    line(df=system_communication_df, 
        x='batch_size', 
        y=['communication_throughput'], 
        title='System Communication Bytes',
        xlabel='Batch Size',
        ylabel='Bytes Per Second',
        legend=['Communication Throughput'],
        path=f'{output}/system_communication_throughput.png')

    # Layer Weights 
    bar(df=system_layer_weight_df, 
        x='batch_size', 
        y=['bottom_mlp_weights', 'embedding_weights', 'interaction_weights', 'top_mlp_weights'], 
        title='System Model Weights', 
        xlabel='Batch Size', 
        ylabel='Bytes', 
        legend=['Bottom MLP', 'Embedding', 'Interaction', 'Top MLP'], 
        path=f'{output}/system_model_weights.png',
        stacked=True)

    # Layer Percentage
    bar(df=system_layer_percentage_df, 
        x='batch_size', 
        y=['server_input_percentage', 'server_bottom_mlp_percentage', 'server_embedding_percentage', 'server_communication_percentage', 'server_interaction_percentage', 'server_top_mlp_percentage', 'server_output_percentage'], 
        title='System Layer Percentage', 
        xlabel='Batch Size', 
        ylabel='Percentage', 
        legend=['Input', 'Bottom MLP', 'Embedding', 'Communication', 'Interaction', 'Top MLP', 'Output'], 
        path=f'{output}/system_layer_percentage.png',
        stacked = True)

    # Layer Percentage without IO
    bar(df=system_layer_percentage_without_io_df, 
        x='batch_size', 
        y=['offline_bottom_mlp_percentage', 'offline_embedding_percentage', 'offline_communication_percentage', 'offline_interaction_percentage', 'offline_top_mlp_percentage'], 
        title='System Layer Percentage Without IO', 
        xlabel='Batch Size', 
        ylabel='Bytes',
        legend=['Bottom MLP', 'Embedding', 'Communication', 'Interaction', 'Top MLP'], 
        path=f'{output}/system_layer_percentage_without_io.png',
        stacked = True)

    # Layer Latency 
    line(df=system_layer_latency_df, 
        x='batch_size', 
        y=['input_device_duration_sum', 'bottom_mlp_device_duration_sum', 'embedding_device_duration_sum', 'interaction_device_duration_sum', 'top_mlp_device_duration_sum', 'output_device_duration_sum'], 
        title='System Layer Latency', 
        xlabel='Batch Size', 
        ylabel='Latency', 
        legend=['Input', 'Bottom MLP', 'Embedding', 'Interaction', 'Top MLP', 'Output'], 
        path=f'{output}/system_layer_latency.png')

    # Layer Latency without IO
    line(df=system_layer_latency_df, 
        x='batch_size', 
        y=['bottom_mlp_device_duration_sum', 'embedding_device_duration_sum', 'interaction_device_duration_sum', 'top_mlp_device_duration_sum'], 
        title='System Layer Latency Without IO', 
        xlabel='Batch Size', 
        ylabel='Bytes', 
        legend=['Bottom MLP', 'Embedding', 'Interaction', 'Top MLP'], 
        path=f'{output}/system_layer_latency_without_io.png')

    # # Worker Layer Percentage in Offline Scenario
    # plt.clf()
    # plt.ylim(0, 1)
    # scale = 6
    # ncol = 3
    # batch_size_list = offline_layer_percentage_df['batch_size'].unique()
    # nrow = numpy.ceil(batch_size_list.size / ncol).astype(int)
    # fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(nrow * scale, ncol * scale))
    # # axes[-1][-1].axis('off')
    # # axes[-1][-2].axis('off')
    # for i, batch_size in enumerate(batch_size_list):
    #     batch_layer_percentage_df = offline_layer_percentage_df[offline_layer_percentage_df['batch_size'] == batch_size]
    #     batch_layer_percentage_df = batch_layer_percentage_df.sort_values('local_rank')
    #     axs = batch_layer_percentage_df.plot(ax = axes[i // ncol][i % ncol], x='local_rank', y=['offline_bottom_mlp_percentage', 'offline_embedding_percentage', 'offline_communication_percentage', 'offline_interaction_percentage', 'offline_top_mlp_percentage'],  kind='bar', grid=True, title=f'Batch {batch_size}', xlabel='Worker', ylabel='Ratio', stacked = True)
    #     axs.legend(['Bottom MLP', 'Embedding', 'Communication', 'Interaction','Top MLP'], loc = 'best',  framealpha=0.5)
    # plt.tight_layout()
    # plt.savefig(f'{input}/offline_layer_percentage.png')

    # # Worker Layer Percentage in Server Scenario
    # plt.clf()
    # plt.ylim(0, 1)
    # scale = 6
    # ncol = 3
    # batch_size_list = server_layer_percentage_df['batch_size'].unique()
    # nrow = numpy.ceil(batch_size_list.size / ncol).astype(int)
    # fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(nrow * scale, ncol * scale))
    # for i, batch_size in enumerate(batch_size_list):
    #     batch_layer_percentage_df = server_layer_percentage_df[server_layer_percentage_df['batch_size'] == batch_size]
    #     batch_layer_percentage_df = batch_layer_percentage_df.sort_values('local_rank')
    #     axs = batch_layer_percentage_df.plot(ax = axes[i // ncol][i % ncol], x='local_rank', y=['server_input_percentage', 'server_bottom_mlp_percentage', 'server_embedding_percentage', 'server_communication_percentage', 'server_interaction_percentage', 'server_top_mlp_percentage', 'server_output_percentage'],  kind='bar', grid=True, title=f'Batch {batch_size}', xlabel='Worker', ylabel='Ratio', stacked = True)
    #     axs.legend(['Input', 'Bottom MLP', 'Embedding', 'Communication', 'Interaction', 'Top MLP', 'Output'], loc = 'best',  framealpha=0.5)
    # plt.tight_layout()
    # plt.savefig(f'{input}/server_layer_percentage.png')


    # # Worker Layer Latency
    # plt.clf()
    # plt.ylim(0, 1)
    # scale = 6
    # ncol = 3
    # batch_size_list = layer_system_latency_df['batch_size'].unique()
    # nrow = numpy.ceil(batch_size_list.size / ncol).astype(int)
    # fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(nrow * scale, ncol * scale))
    # for i, batch_size in enumerate(batch_size_list):
    #     batch_layer_percentage_df = layer_system_latency_df[layer_system_latency_df['batch_size'] == batch_size]
    #     batch_layer_percentage_df = batch_layer_percentage_df.sort_values('local_rank')
    #     axs = batch_layer_percentage_df.plot(ax = axes[i // ncol][i % ncol], x='local_rank', y=['input_device_duration', 'bottom_mlp_device_duration', 'embedding_device_duration', 'communication_transfer_device_duration', 'interaction_device_duration', 'top_mlp_device_duration', 'output_device_duration'],  kind='bar', grid=True, title=f'Batch {batch_size}', xlabel='Worker', ylabel='Latency', stacked = False)
    #     axs.legend(['Input', 'Bottom MLP', 'Embedding', 'Communication', 'Interaction', 'Top MLP', 'Output'], loc = 'best',  framealpha=0.5)
    # plt.tight_layout()
    # plt.savefig(f'{input}/layer_latency.png')

    # # Worker Compute Layer Latency
    # plt.clf()
    # plt.ylim(0, 1)
    # scale = 6
    # ncol = 3
    # batch_size_list = layer_system_latency_df['batch_size'].unique()
    # nrow = numpy.ceil(batch_size_list.size / ncol).astype(int)
    # fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(nrow * scale, ncol * scale))
    # for i, batch_size in enumerate(batch_size_list):
    #     batch_layer_percentage_df = layer_system_latency_df[layer_system_latency_df['batch_size'] == batch_size]
    #     batch_layer_percentage_df = batch_layer_percentage_df.sort_values('local_rank')
    #     axs = batch_layer_percentage_df.plot(ax = axes[i // ncol][i % ncol], x='local_rank', y=['bottom_mlp_device_duration', 'embedding_device_duration', 'interaction_device_duration', 'top_mlp_device_duration'],  kind='bar', grid=True, title=f'Batch {batch_size}', xlabel='Worker', ylabel='Latency', stacked = False)
    #     axs.legend(['Bottom MLP', 'Embedding', 'Interaction', 'Top MLP'], loc = 'best',  framealpha=0.5)
    # plt.tight_layout()
    # plt.savefig(f'{input}/compute_layer_latency.png')

    return 0

exit(main())