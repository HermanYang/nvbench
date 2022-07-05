import argparse
import sqlite3
import pathlib
import pandas 
import matplotlib.pyplot as plt
import numpy
import json
from multiprocessing import Pool
from shutil import rmtree
from os import mkdir

parser = argparse.ArgumentParser()

parser.add_argument('--input', type=str, required=True, help='Path to nvbench results')
parser.add_argument('--output', type=str, required=True, help='Path analysis results')
parser.add_argument('--override', action='store_true', help='Override existing results')

def set_global_plot_params(figure_side_len = 8, font_size = 8):
    golden_ratio = (1 + 5 ** 0.5) / 2
    side_len = figure_side_len
    fontsize = font_size * golden_ratio
    fontsize_big = fontsize *  golden_ratio
    plt.rc('figure', figsize = (side_len * golden_ratio, side_len))
    plt.rc('font', size = fontsize)          # controls default text sizes
    plt.rc('axes', titlesize = fontsize_big)     # fontsize of the axes title
    plt.rc('axes', labelsize = fontsize_big)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize = fontsize)    # fontsize of the tick labels
    plt.rc('ytick', labelsize = fontsize)    # fontsize of the tick labels
    plt.rc('legend', fontsize = fontsize_big)    # legend fontsize
    plt.rc('figure', titlesize = fontsize_big)  # fontsize of the figure title


def df_draw(df, ax, kind, x, y, title, xlabel, ylabel, legend, logx = True, logy = True):
    origin_df = df
    df=df.copy(deep=True)

    if logx:
        df[x] = df[x].transform(numpy.log2)

    if kind == 'line':
        ax = df.plot(ax = ax, x=x, y=y, kind='line', grid=True, marker = 'o')
    elif kind == 'bar':
        ax = df.plot(ax = ax, x=x, y=y, kind='bar', grid=True, stacked = False)
    elif kind == 'stacked_bar':
        ax = df.plot(ax = ax, x=x, y=y, kind='bar', grid=True, stacked = True)

    ax.set_title(label = title)
    ax.set_xlabel(xlabel = xlabel)
    ax.set_ylabel(ylabel = ylabel)

    if logx:
        ax.set_xticks(df[x].to_numpy())
        ax.set_xticklabels(labels = origin_df[x].to_numpy().astype(str))

    if logy:
        ax.set_yscale('log', base=10)
    ax.legend(legend, loc = 'best', framealpha=0.5)

    return ax

def plot(df, kind, x, y, title, xlabel, ylabel, legend, logx = True, logy = True):
    fig, ax = plt.subplots()
    df_draw(df, ax, kind, x, y, title, xlabel, ylabel, legend, logx, logy)
    fig.tight_layout()
    return fig


def plots(df, key, kind, x, y, title, ylabel, legend, logx = True, logy = True):
    ncol = 3
    key_list = df[key].unique()
    nrow = numpy.ceil(key_list.size / ncol).astype(int)
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol)
    for i, key_value in enumerate(key_list):
        batch_df = df[df[key] == key_value]
        batch_df = batch_df.sort_values(x)
        xlabel_tokens = f'{key} {key_value}'.replace('_', ' ').split()
        xlabel = ' '.join(map(lambda x: x.capitalize(), xlabel_tokens))
        df_draw(df=batch_df,
            ax = axes[i // ncol][i % ncol],
            kind = kind,
            x = x,
            y = y,
            title = title,
            xlabel = xlabel,
            ylabel = ylabel,
            legend = legend,
            logx = logx,
            logy = logy)

    if nrow * ncol > len(key_list):
        for i in range(len(key_list), nrow * ncol):
            axes[i // ncol][i % ncol].axis('off')

    fig.tight_layout()
    return fig


def analyze_parallel(paths):
    return analyze(paths['nsysdb_path'], paths['power_draw_path'])


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

    nanosecond_to_second = 1e-9
    input_stats_df = pandas.read_sql_query('SELECT * FROM step_input_stats', con)
    output_stats_df = pandas.read_sql_query('SELECT * FROM step_output_stats', con)
    compute_stats_df = pandas.read_sql_query('SELECT * FROM step_compute_stats', con)
    comunication_wait_stats_df = pandas.read_sql_query('SELECT * FROM step_communication_wait_stats', con)
    comunication_transfer_stats_df = pandas.read_sql_query('SELECT * FROM step_communication_transfer_stats', con)
    comunication_transfer_stats_df = comunication_transfer_stats_df.assign(throughput = lambda stats: (stats['send_bytes'] + stats['recv_bytes']) / (stats['device_duration'] * nanosecond_to_second) )
    inference_stats_df = pandas.read_sql_query('SELECT * FROM step_inference_stats', con)

    cur.close()

    # merge all stats
    worker_stats_df = inference_stats_df.merge(
        input_stats_df.rename(columns={'host_duration':'input_host_duration', 'device_duration':'input_device_duration', 'host_bytes':'input_host_bytes', 'device_bytes':'input_device_bytes', 'throughput':'input_throughput'}), 
        how='left', on=['rank', 'local_rank', 'globalTid','step_id'])
    worker_stats_df = worker_stats_df.merge(comunication_wait_stats_df.rename(columns={'host_duration':'communication_host_duration', 'device_duration':'communication_wait_device_duration', 'send_bytes':'communication_send_bytes', 'recv_bytes':'communication_recv_bytes'}), how='left', on=['rank', 'local_rank', 'globalTid', 'step_id'])
    worker_stats_df = worker_stats_df.merge(comunication_transfer_stats_df.filter(['rank', 'local_rank', 'globalTid','step_id', 'device_duration', 'throughput']).rename(columns={'device_duration':'communication_transfer_device_duration', 'throughput':'communication_throughput'}), how='left', on=['rank', 'local_rank', 'globalTid', 'step_id'])
    bottom_mlp_stats_df = compute_stats_df[compute_stats_df['event_value'] == 'bottom_mlp'].filter(items=['step_id', 'rank', 'local_rank', 'globalTid', 'weights', 'host_duration', 'device_duration'])
    embedding_stats_df = compute_stats_df[compute_stats_df['event_value'] == 'embedding'].filter(items=['step_id', 'rank', 'local_rank', 'globalTid', 'weights', 'host_duration', 'device_duration'])
    interaction_stats_df = compute_stats_df[compute_stats_df['event_value'] == 'interaction'].filter(items=['step_id', 'rank', 'local_rank', 'globalTid', 'weights', 'host_duration', 'device_duration'])
    top_mlp_stats_df = compute_stats_df[compute_stats_df['event_value'] == 'top_mlp'].filter(items=['step_id', 'rank', 'local_rank', 'globalTid', 'weights', 'host_duration', 'device_duration'])

    worker_stats_df = worker_stats_df.merge(bottom_mlp_stats_df.rename(columns={'weights':'bottom_mlp_weights', 'host_duration':'bottom_mlp_host_duration', 'device_duration':'bottom_mlp_device_duration'}), how='left',  on=['rank', 'local_rank','globalTid','step_id'])
    worker_stats_df = worker_stats_df.merge(embedding_stats_df.rename(columns={'weights':'embedding_weights', 'host_duration':'embedding_host_duration', 'device_duration':'embedding_device_duration'}), how='left', on=['rank', 'local_rank','globalTid','step_id'])
    worker_stats_df = worker_stats_df.merge(interaction_stats_df.rename(columns={'weights':'interaction_weights', 'host_duration':'interaction_host_duration', 'device_duration':'interaction_device_duration'} ), how='left', on=['rank', 'local_rank','globalTid','step_id'])
    worker_stats_df = worker_stats_df.merge(top_mlp_stats_df.rename(columns={'weights':'top_mlp_weights', 'host_duration':'top_mlp_host_duration', 'device_duration':'top_mlp_device_duration'}), how='left', suffixes=['', '_top_mlp'], on=['rank', 'local_rank','globalTid','step_id'])
    worker_stats_df = worker_stats_df.merge(output_stats_df.rename(columns={'host_duration':'output_host_duration', 'device_duration':'output_device_duration', 'host_bytes':'output_host_bytes', 'device_bytes':'output_device_bytes', 'throughput':'output_throughput'}), how='left', on=['rank', 'local_rank','globalTid','step_id'])
    worker_stats_df = worker_stats_df.fillna(0)

    worker_stats_df = worker_stats_df.assign(device_duration = lambda stats: stats['input_device_duration'] + stats['bottom_mlp_device_duration'] + stats['embedding_device_duration'] + stats['communication_wait_device_duration'] + stats['communication_transfer_device_duration'] + stats['interaction_device_duration'] + stats['top_mlp_device_duration'] + stats['output_device_duration'])
    # estimate device_duration_without_wait, only consider input + compute + communication + output time on device
    worker_stats_df = worker_stats_df.assign(device_duration_without_wait = lambda stats: stats['input_device_duration'] + stats['bottom_mlp_device_duration'] + stats['embedding_device_duration'] + stats['communication_transfer_device_duration'] + stats['interaction_device_duration'] + stats['top_mlp_device_duration'] + stats['output_device_duration'])
    # estimate device_duration_without_wait_and_io, only consider compute + communication on device
    worker_stats_df = worker_stats_df.assign(device_duration_without_wait_and_io = lambda stats: stats['bottom_mlp_device_duration'] + stats['embedding_device_duration'] + stats['communication_transfer_device_duration'] + stats['interaction_device_duration'] + stats['top_mlp_device_duration'])

    # server scenatio layer ratio
    ratio_to_percentage=100
    worker_stats_df = worker_stats_df.assign(server_input_percentage = lambda stats: (stats['input_device_duration'] / stats['device_duration_without_wait']) * ratio_to_percentage)
    worker_stats_df = worker_stats_df.assign(server_output_percentage = lambda stats: (stats['output_device_duration'] / stats['device_duration_without_wait']) * ratio_to_percentage)
    worker_stats_df = worker_stats_df.assign(server_bottom_mlp_percentage = lambda stats: (stats['bottom_mlp_device_duration'] / stats['device_duration_without_wait']) * ratio_to_percentage)
    worker_stats_df = worker_stats_df.assign(server_embedding_percentage = lambda stats: (stats['embedding_device_duration'] / stats['device_duration_without_wait']) * ratio_to_percentage)
    worker_stats_df = worker_stats_df.assign(server_communication_percentage = lambda stats: (stats['communication_transfer_device_duration'] / stats['device_duration_without_wait']) * ratio_to_percentage)
    worker_stats_df = worker_stats_df.assign(server_interaction_percentage = lambda stats: (stats['interaction_device_duration'] / stats['device_duration_without_wait']) * ratio_to_percentage)
    worker_stats_df = worker_stats_df.assign(server_top_mlp_percentage = lambda stats: (stats['top_mlp_device_duration'] / stats['device_duration_without_wait']) * ratio_to_percentage)

    # offline scenatio layer ratio
    worker_stats_df = worker_stats_df.assign(offline_bottom_mlp_percentage = lambda stats: (stats['bottom_mlp_device_duration'] / stats['device_duration_without_wait_and_io']) * ratio_to_percentage)
    worker_stats_df = worker_stats_df.assign(offline_embedding_percentage = lambda stats: (stats['embedding_device_duration'] / stats['device_duration_without_wait_and_io']) * ratio_to_percentage)
    worker_stats_df = worker_stats_df.assign(offline_communication_percentage = lambda stats: (stats['communication_transfer_device_duration'] / stats['device_duration_without_wait_and_io']) * ratio_to_percentage)
    worker_stats_df = worker_stats_df.assign(offline_interaction_percentage = lambda stats: (stats['interaction_device_duration'] / stats['device_duration_without_wait_and_io']) * ratio_to_percentage)
    worker_stats_df = worker_stats_df.assign(offline_top_mlp_percentage = lambda stats: (stats['top_mlp_device_duration'] / stats['device_duration_without_wait_and_io']) * ratio_to_percentage)

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
        device_duration =('device_duration', 'max'), 
        device_duration_sum =('device_duration', 'sum'), 
        device_duration_without_wait=('device_duration_without_wait', 'max'), 
        device_duration_without_wait_sum =('device_duration_without_wait', 'sum'), 
        device_duration_without_wait_and_io=('device_duration_without_wait_and_io', 'max'), 
        device_duration_without_wait_and_io_sum=('device_duration_without_wait_and_io', 'sum'))
    
    # system level layer percentage
    system_stats_df = system_stats_df.assign(server_input_percentage = lambda stats: (stats['input_device_duration_sum'] / stats['device_duration_without_wait_sum']) * ratio_to_percentage)
    system_stats_df = system_stats_df.assign(server_output_percentage = lambda stats: (stats['output_device_duration_sum'] / stats['device_duration_without_wait_sum']) * ratio_to_percentage)
    system_stats_df = system_stats_df.assign(server_bottom_mlp_percentage = lambda stats: (stats['bottom_mlp_device_duration_sum'] / stats['device_duration_without_wait_sum']) * ratio_to_percentage)
    system_stats_df = system_stats_df.assign(server_embedding_percentage = lambda stats: (stats['embedding_device_duration_sum'] / stats['device_duration_without_wait_sum']) * ratio_to_percentage)
    system_stats_df = system_stats_df.assign(server_communication_percentage = lambda stats: (stats['communication_transfer_device_duration_sum'] / stats['device_duration_without_wait_sum']) * ratio_to_percentage)
    system_stats_df = system_stats_df.assign(server_interaction_percentage = lambda stats: (stats['interaction_device_duration_sum'] / stats['device_duration_without_wait_sum']) * ratio_to_percentage)
    system_stats_df = system_stats_df.assign(server_top_mlp_percentage = lambda stats: (stats['top_mlp_device_duration_sum'] / stats['device_duration_without_wait_sum']) * ratio_to_percentage)

    system_stats_df = system_stats_df.assign(offline_input_percentage = lambda stats: (stats['input_device_duration_sum'] / stats['device_duration_without_wait_and_io_sum']) * ratio_to_percentage)
    system_stats_df = system_stats_df.assign(offline_output_percentage = lambda stats: (stats['output_device_duration_sum'] / stats['device_duration_without_wait_and_io_sum']) * ratio_to_percentage)
    system_stats_df = system_stats_df.assign(offline_bottom_mlp_percentage = lambda stats: (stats['bottom_mlp_device_duration_sum'] / stats['device_duration_without_wait_and_io_sum']) * ratio_to_percentage)
    system_stats_df = system_stats_df.assign(offline_embedding_percentage = lambda stats: (stats['embedding_device_duration_sum'] / stats['device_duration_without_wait_and_io_sum']) * ratio_to_percentage)
    system_stats_df = system_stats_df.assign(offline_communication_percentage = lambda stats: (stats['communication_transfer_device_duration_sum'] / stats['device_duration_without_wait_and_io_sum']) * ratio_to_percentage)
    system_stats_df = system_stats_df.assign(offline_interaction_percentage = lambda stats: (stats['interaction_device_duration_sum'] / stats['device_duration_without_wait_and_io_sum']) * ratio_to_percentage)
    system_stats_df = system_stats_df.assign(offline_top_mlp_percentage = lambda stats: (stats['top_mlp_device_duration_sum'] / stats['device_duration_without_wait_and_io_sum']) * ratio_to_percentage)

    system_stats_df = system_stats_df.assign(host_duration = lambda stats: stats['host_end'] - stats['host_start'])
    system_stats_df = system_stats_df.drop(columns=['host_start', 'host_end'])

    system_stats_df = system_stats_df.assign(input_throughput = lambda stats: stats['input_bytes'] / (stats['input_device_duration_sum'] * nanosecond_to_second))
    system_stats_df = system_stats_df.assign(output_throughput = lambda stats: stats['output_bytes'] / (stats['output_device_duration_sum'] * nanosecond_to_second))
    system_stats_df = system_stats_df.assign(communication_throughput = lambda stats: (stats['send_bytes'] + stats['recv_bytes']) / (stats['communication_transfer_device_duration_sum'] * nanosecond_to_second))
    system_stats_df = system_stats_df.assign(ips = lambda stats: stats['batch_size'] / (stats['host_duration'] * nanosecond_to_second))
    system_stats_df = system_stats_df.assign(ideal_ips = lambda stats: stats['batch_size'] / (stats['device_duration'] * nanosecond_to_second))
    system_stats_df = system_stats_df.assign(ideal_ips_without_wait = lambda stats: stats['batch_size'] / (stats['device_duration_without_wait'] * nanosecond_to_second))
    system_stats_df = system_stats_df.assign(ideal_ips_without_wait_and_io = lambda stats: stats['batch_size'] / (stats['device_duration_without_wait_and_io'] * nanosecond_to_second))

    # power draw estimation with real ips
    system_stats_df = system_stats_df.assign(ipw = lambda stats: stats['ips'] / sum(worker_power_draw))

    return worker_stats_df, system_stats_df


def main():
    args = parser.parse_args()

    input = pathlib.Path(args.input)
    output = pathlib.Path(args.output)
    override = args.override

    if not input.exists():
        raise ValueError('Input does not exist: {}'.format(input))

    if output.exists():
        if override:
            rmtree(output)
        else:
            raise ValueError('Output {} exist, add --override to enable override'.format(output))

    mkdir(path=output)

    system_stats_df = pandas.DataFrame()
    worker_stats_df = pandas.DataFrame()

    inputs = []
    for file in input.glob('**/*.sqlite'):
        nsysdb_path = f'{input}/{file.stem}.sqlite'
        power_draw_path = f'{input}/{file.stem}_power_draw.csv'

        if not pathlib.Path(nsysdb_path).exists():
            raise ValueError('Path does not exist: {}'.format(nsysdb_path))

        if not pathlib.Path(power_draw_path).exists():
            raise ValueError('Path does not exist: {}'.format(power_draw_path))

        inputs.append({'nsysdb_path':nsysdb_path, 'power_draw_path':power_draw_path})
    
    pool = Pool(processes=len(inputs))
    outputs = pool.map(analyze_parallel, inputs)

    for pair in outputs:
        worker_stats_df = pandas.concat([worker_stats_df, pair[0]])
        system_stats_df = pandas.concat([system_stats_df, pair[1]])

    worker_stats_df.to_csv(f'{output}/worker_stats.csv')
    system_stats_df.to_csv(f'{output}/system_stats.csv')

    # System Level Analysis
    set_global_plot_params(figure_side_len=8, font_size=8)
    # IPS
    items=['batch_size', 'ips', 'ideal_ips', 'ideal_ips_without_wait', 'ideal_ips_without_wait_and_io']
    header =['Batch Size', 'IPS', 'Ideal IPS', 'Ideal IPS without Wait', 'Ideal IPS without Wait and IO']
    system_ips_df = system_stats_df[system_stats_df.index == system_stats_df.index.max()].filter(items=items).sort_values(items[0]).reset_index(drop=True)
    system_ips_df.to_csv(f'{output}/system_ips.csv', header = header, index=False)
    plt.clf()
    plot(df=system_ips_df, 
        kind='line',
        x=items[0],
        y=items[1:],
        title='System Throughput', 
        xlabel=header[0], 
        ylabel='Inference Per Second', 
        legend=header[1:], 
        logy = True).savefig(f'{output}/system_ips.png')

    # IPW
    items=['batch_size', 'ipw']
    header=['Batch Size', 'IPW']
    system_ipw_df = system_stats_df[system_stats_df.index == system_stats_df.index.max()].filter(items=items).sort_values(items[0]).reset_index(drop=True)
    system_ipw_df.to_csv(f'{output}/system_ipw.csv', header = header, index=False)
    plot(df=system_ipw_df,
        kind='line',
        x=items[0], 
        y=items[1:],
        title='System Energey Efficency', 
        xlabel=header[0], 
        ylabel='Inference Per Watt', 
        legend=header[1:], 
        logy = True).savefig(f'{output}/system_ipw.png')

    # E2E Latency
    items=['batch_size', 'host_duration', 'device_duration', 'device_duration_without_wait', 'device_duration_without_wait_and_io']
    header=['Batch Size', 'Latency(ns)', 'Device Latency(ns)', 'Device Latency Without Wait(ns)', 'Device Latency Without Wait and IO(ns)']
    system_latency_df = system_stats_df[system_stats_df.index == system_stats_df.index.max()].filter(items = items).sort_values(items[0]).reset_index(drop=True)
    system_latency_df.to_csv(f'{output}/system_e2e_latency.csv', header = header, index=False)
    plot(df=system_latency_df, 
        kind='line',
        x= items[0],
        y= items[1:], 
        title='System E2E Latency', 
        xlabel=header[0], 
        ylabel='Latency(ns)', 
        legend=header[1:], 
        logy = True).savefig(f'{output}/system_e2e_latency.png')

    # IO bytes
    items=['batch_size', 'input_bytes', 'output_bytes']
    header=['Batch Size', 'Input(bytes)', 'Output(bytes)']
    system_io_bytes_df = system_stats_df[system_stats_df.index == system_stats_df.index.max()].filter(items=items).sort_values(items[0]).reset_index(drop=True)
    system_io_bytes_df.to_csv(f'{output}/system_io.csv', header = header, index=False)
    plot(df=system_io_bytes_df, 
        kind='stacked_bar',
        x=items[0], 
        y=items[1:],
        title='System IO Bytes',
        xlabel=header[0], 
        ylabel='Bytes', 
        legend=header[1:], 
        logy = False).savefig(f'{output}/system_io_bytes.png')

    # IO Throughput 
    items=['batch_size', 'input_throughput', 'output_throughput']
    header=['Batch Size', 'Input(bytes/s)', 'Output(bytes/s)']
    system_io_throughput_df = system_stats_df[system_stats_df.index == system_stats_df.index.max()].filter(items = items).sort_values(items[0]).reset_index(drop=True)
    system_io_throughput_df.to_csv(f'{output}/system_io_throughput.csv', header = header, index=False)
    plt.clf()
    plot(df=system_io_throughput_df, 
        kind='line',
        x='batch_size', 
        y=['input_throughput', 'output_throughput'], 
        title='System IO Throughput', 
        xlabel=header[0], 
        ylabel='Bytes Per Second', 
        legend=header[1:], 
        logy = True)
    plt.savefig(f'{output}/system_io_throughput.png')

    # Communication bytes
    items=['batch_size', 'send_bytes', 'recv_bytes']
    header = ['Batch Size', 'Send(bytes)', 'Receive(bytes)']
    system_communication_bytes_df = system_stats_df[system_stats_df.index == system_stats_df.index.max()].filter(items=items).sort_values(items[0]).reset_index(drop=True)
    system_communication_bytes_df.to_csv(f'{output}/system_communication_bytes.csv', header = header, index=False)
    plt.clf()
    plot(df=system_communication_bytes_df, 
        kind='stacked_bar',
        x=items[0], 
        y=items[1:], 
        title='System Communication Bytes', 
        xlabel=header[0], 
        ylabel='Bytes', 
        legend=header[1:],
        logy = False)
    plt.savefig(f'{output}/system_communication_bytes.png')

    # Communication Throughtput
    items=['batch_size', 'communication_throughput']
    header = ['Batch Size', 'Throughput(bytes/s)']
    system_communication_throughput_df = system_stats_df[system_stats_df.index == system_stats_df.index.max()].filter(items = items).sort_values(items[0]).reset_index(drop=True)
    system_communication_throughput_df.to_csv(f'{output}/system_communication_throughput.csv', header = header, index=False)
    plot(df=system_communication_throughput_df, 
        kind='line',
        x=items[0], 
        y=items[1:], 
        title='System Communication Throughput',
        xlabel=header[0], 
        ylabel='Bytes Per Second',
        legend=header[1:],
        logy = True).savefig(f'{output}/system_communication_throughput.png')

    # Layer Weights 
    items=['batch_size', 'bottom_mlp_weights', 'embedding_weights', 'interaction_weights', 'top_mlp_weights']
    header = ['Batch Size', 'Bottom MLP(bytes)', 'Embedding(bytes)', 'Interaction(bytes)', 'Top MLP(bytes)']
    system_layer_weight_df = system_stats_df[system_stats_df.index == system_stats_df.index.max()].filter(items = items).sort_values(items[0]).reset_index(drop=True)
    system_layer_weight_df.to_csv(f'{output}/system_layer_weights.csv', header = header, index=False)
    plot(df=system_layer_weight_df, 
        kind='stacked_bar',
        x='batch_size', 
        y=['bottom_mlp_weights', 'embedding_weights', 'interaction_weights', 'top_mlp_weights'], 
        title='System Model Weights', 
        xlabel=header[0], 
        ylabel='Bytes', 
        legend=header[1:], 
        logy = False).savefig(f'{output}/system_model_weights.png')

    # Layer Latency 
    items=['batch_size', 'input_device_duration_sum', 'bottom_mlp_device_duration_sum', 'embedding_device_duration_sum', 'communication_transfer_device_duration_sum', 'interaction_device_duration_sum', 'top_mlp_device_duration_sum', 'output_device_duration_sum']
    header = ['Batch Size', 'Input(ns)', 'Bottom MLP(ns)', 'Embedding(ns)', 'Communication(ns)', 'Interaction(ns)', 'Top MLP(ns)', 'Output(ns)']
    system_layer_latency_without_communication_wait_df = system_stats_df[system_stats_df.index == system_stats_df.index.max()].filter(items = items).sort_values(items[0]).reset_index(drop=True)
    system_layer_latency_without_communication_wait_df.to_csv(f'{output}/system_layer_latency_without_communication_wait.csv', header = header, index=False)
    plot(df=system_layer_latency_without_communication_wait_df, 
        kind='stacked_bar',
        x=items[0], 
        y=items[1:], 
        title='System Layer Latency without Communication Wait', 
        xlabel=header[0], 
        ylabel='Latency', 
        legend = header[1:],
        logy = False).savefig(f'{output}/system_layer_latency.png')

    # Layer Latency without IO
    items = ['batch_size', 'bottom_mlp_device_duration_sum', 'embedding_device_duration_sum', 'communication_transfer_device_duration_sum', 'interaction_device_duration_sum', 'top_mlp_device_duration_sum']
    header = ['Batch Size',  'Bottom MLP(ns)', 'Embedding(ns)', 'Communication(ns)', 'Interaction(ns)', 'Top MLP(ns)']
    system_layer_latency_without_communication_wait_and_io_df = system_stats_df[system_stats_df.index == system_stats_df.index.max()].filter(items = items).sort_values(items[0]).reset_index(drop=True)
    system_layer_latency_without_communication_wait_and_io_df.to_csv(f'{output}/system_layer_latency_without_communication_wait_and_io.csv', header = header, index=False)
    plot(df=system_layer_latency_without_communication_wait_and_io_df, 
        kind='stacked_bar',
        x=items[0], 
        y=items[1:],
        title='System Layer Latency Without Communication Wait and IO', 
        xlabel=header[0], 
        ylabel='Bytes', 
        legend=header[1:],
        logy = False).savefig(f'{output}/system_layer_latency_without_io.png')

    # Layer Percentage
    items=['batch_size', 'server_input_percentage','server_bottom_mlp_percentage', 'server_embedding_percentage', 'server_communication_percentage', 'server_interaction_percentage', 'server_top_mlp_percentage', 'server_output_percentage']
    header =['Batch Size', 'Input(%)', 'Bottom MLP(%)', 'Embedding(%)', 'Communication(%)', 'Interaction(%)', 'Top MLP(%)', 'Output(%)']
    system_layer_percentage_without_communication_wait_df = system_stats_df[system_stats_df.index == system_stats_df.index.max()].filter(items = items).sort_values(items[0]).reset_index(drop=True)
    system_layer_percentage_without_communication_wait_df.to_csv(f'{output}/system_layer_percentage_without_communication_wait.csv', header = header, index=False) 
    plot(df=system_layer_percentage_without_communication_wait_df, 
        kind='stacked_bar',
        x=items[0], 
        y=items[1:],
        title='System Layer Percentage Without Communication Wait', 
        xlabel=header[0], 
        ylabel='Percentage', 
        legend = header[1:],
        logy = False).savefig(f'{output}/system_layer_percentage.png')

    # Layer Percentage without IO
    items=['batch_size', 'offline_bottom_mlp_percentage', 'offline_embedding_percentage', 'offline_communication_percentage', 'offline_interaction_percentage', 'offline_top_mlp_percentage']
    header =['Batch Size', 'Bottom MLP(%)', 'Embedding(%)', 'Communication(%)', 'Interaction(%)', 'Top MLP(%)']
    system_layer_percentage_without_communication_wait_and_io_df = system_stats_df[system_stats_df.index == system_stats_df.index.max()].filter(items = items).sort_values(items[0]).reset_index(drop=True)
    system_layer_percentage_without_communication_wait_and_io_df.to_csv(f'{output}/system_layer_percentage_without_communication_wait_and_io.csv', header = header, index=False)
    plot(df=system_layer_percentage_without_communication_wait_and_io_df, 
        kind='stacked_bar',
        x=items[0], 
        y=items[1:],
        title='System Layer Percentage Without Communication Wait and IO', 
        xlabel=header[0], 
        ylabel='Bytes',
        legend = header[1:],
        logy = False).savefig(f'{output}/system_layer_percentage_without_io.png')

    # Worker Level Analysis
    set_global_plot_params(figure_side_len=32, font_size=8)
    # E2E Latency
    items=['batch_size', 'rank', 'local_rank', 'host_duration', 'device_duration', 'device_duration_without_wait', 'device_duration_without_wait_and_io']
    worker_e2e_latency_df = worker_stats_df[worker_stats_df['step_id'] == worker_stats_df['step_id'].max()].filter(items = items).sort_values(items[0]).reset_index(drop=True)
    plots(df = worker_e2e_latency_df, 
        key = 'batch_size',
        kind = 'bar',
        x = 'rank',
        y = ['host_duration', 'device_duration', 'device_duration_without_wait', 'device_duration_without_wait_and_io'],
        title = 'Worker E2E Latency',
        ylabel = 'Latency(ns)',
        legend=['Host Latency(ns)', 'Device Latency(ns)', 'Device Latency Without Wait(ns)', 'Device Latency Without Wait And IO(ns)'],
        logx = False,
        logy = False).savefig(f'{output}/worker_e2e_latency.png')

    # Input Ouput Bytes
    worker_io_df = worker_stats_df[worker_stats_df['step_id'] == worker_stats_df['step_id'].max()].filter(items=['batch_size', 'rank', 'local_rank', 'input_device_duration', 'input_device_bytes', 'input_throughput', 'output_device_duration', 'output_device_bytes', 'output_throughput']).sort_values('batch_size').reset_index(drop=True)
    plots(df=worker_io_df,
        key = 'batch_size',
        kind = 'stacked_bar',
        x = 'rank',
        y = ['input_device_bytes', 'output_device_bytes'],
        title = 'Worker IO Bytes',
        ylabel = 'Bytes',
        legend=['Input(bytes)', 'Output(bytes)'],
        logx = False,
        logy = False).savefig(f'{output}/worker_io_bytes.png')

    # Input Ouput Throughput
    plots(df=worker_io_df,
        key = 'batch_size',
        kind = 'stacked_bar',
        x = 'rank',
        y = ['input_throughput', 'output_throughput'],
        title = 'Worker IO Throughput',
        ylabel = 'Bytes',
        legend=['Input(bytes/s)', 'Output(bytes/s)'],
        logx = False,
        logy = False).savefig(f'{output}/worker_io_throughput.png')

    # Communication Bytes
    worker_communication_df = worker_stats_df[worker_stats_df['step_id'] == worker_stats_df['step_id'].max()].filter(items=['batch_size', 'rank', 'local_rank', 'communication_send_bytes', 'communication_recv_bytes', 'communication_throughput']).sort_values('batch_size').reset_index(drop=True)
    plots(df=worker_communication_df,
        key = 'batch_size',
        kind = 'stacked_bar',
        x = 'rank',
        y = ['communication_send_bytes', 'communication_recv_bytes'],
        title = 'Worker Communication Bytes',
        ylabel = 'Bytes',
        legend=['Send(bytes)', 'Receive(bytes)'],
        logx = False,
        logy = False).savefig(f'{output}/worker_communication_bytes.png')

    # Communication Throughput
    plots(df=worker_communication_df,
        key = 'batch_size',
        kind = 'stacked_bar',
        x = 'rank',
        y = ['communication_throughput'],
        title = 'Worker Communication Throughput',
        ylabel = 'Bytes',
        legend=['Throughput(bytes/s)'],
        logx = False,
        logy = False).savefig(f'{output}/worker_communication_throughput.png')

    # Layer Latency
    worker_layer_latency_df = worker_stats_df[worker_stats_df['step_id'] == worker_stats_df['step_id'].max()].filter(items=['batch_size', 'rank', 'local_rank', 'input_device_duration', 'bottom_mlp_device_duration', 'embedding_device_duration', 'communication_wait_device_duration', 'communication_transfer_device_duration', 'interaction_device_duration', 'top_mlp_device_duration', 'output_device_duration']).sort_values('batch_size').reset_index(drop=True)
    plots(df=worker_layer_latency_df,
        key = 'batch_size',
        kind = 'stacked_bar',
        x = 'rank',
        y = ['input_device_duration', 'bottom_mlp_device_duration', 'embedding_device_duration', 'communication_wait_device_duration', 'communication_transfer_device_duration', 'interaction_device_duration', 'top_mlp_device_duration', 'output_device_duration'],
        title = 'Worker Layer Latency',
        ylabel = 'Latency',
        legend=['Input(ns)', 'Bottom MLP(ns)', 'Embedding(ns)', 'Wait(ns)', 'Communication(ns)', 'Interaction(ns)', 'Top MLP(ns)', 'Output(ns)'],
        logx = False,
        logy = False).savefig(f'{output}/worker_layer_latency.png')

    # Layer Latency Without Communication wait
    plots(df=worker_layer_latency_df,
        key = 'batch_size',
        kind = 'stacked_bar',
        x = 'rank',
        y = ['input_device_duration', 'bottom_mlp_device_duration', 'embedding_device_duration', 'communication_transfer_device_duration', 'interaction_device_duration', 'top_mlp_device_duration', 'output_device_duration'],
        title = 'Worker Layer Latency Without Communication Wait',
        ylabel = 'Latency(ns)',
        legend=['Input(ns)', 'Bottom MLP(ns)', 'Embedding(ns)', 'Communication(ns)', 'Interaction(ns)', 'Top MLP(ns)', 'Output(ns)'],
        logx = False,
        logy = False).savefig(f'{output}/worker_layer_latency_without_communication_wait.png')

    # Layer Latency Without Communication wait and IO
    plots(df=worker_layer_latency_df,
        key = 'batch_size',
        kind = 'stacked_bar',
        x = 'rank',
        y = ['bottom_mlp_device_duration', 'embedding_device_duration', 'communication_transfer_device_duration', 'interaction_device_duration', 'top_mlp_device_duration'],
        title = 'Worker Layer Latency Without Communication Wait and IO',
        ylabel = 'Latency(ns)',
        legend=['Bottom MLP(ns)', 'Embedding(ns)', 'Wait(ns)', 'Communication(ns)', 'Interaction(ns)', 'Top MLP(ns)'],
        logx = False,
        logy = False).savefig(f'{output}/worker_layer_latency_without_communication_wait_and_io.png')

    # Layer Percentage
    worker_layer_server_percentage_df = worker_stats_df[worker_stats_df['step_id'] == worker_stats_df['step_id'].max()].filter(items=['batch_size', 'rank', 'local_rank', 'server_input_percentage', 'server_bottom_mlp_percentage', 'server_embedding_percentage', 'server_communication_percentage', 'server_interaction_percentage', 'server_top_mlp_percentage', 'server_output_percentage']).sort_values('batch_size').reset_index(drop=True)
    worker_layer_server_percentage_df.to_csv(f'{output}/server_layer_percentage.csv', header=['Batch Size', 'Rank', 'Local Rank', 'Input(%)', 'Bottom MLP(%)', 'Embedding(%)', 'Communication(%)', 'Interaction(%)', 'Top MLP(%)', 'Output(%)'],index=False)
    plots(df=worker_layer_server_percentage_df,
        key = 'batch_size',
        kind = 'stacked_bar',
        x = 'rank',
        y = ['server_input_percentage', 'server_bottom_mlp_percentage', 'server_embedding_percentage', 'server_communication_percentage', 'server_interaction_percentage', 'server_top_mlp_percentage', 'server_output_percentage'],
        title = 'Worker Layer Percentage Without Communication Wait',
        ylabel = 'Latency(ns)',
        legend=['Input(%)', 'Bottom MLP(%)', 'Embedding(%)', 'Communication(%)', 'Interaction(%)', 'Top MLP(%)', 'Output(%)'],
        logx = False,
        logy = False).savefig(f'{output}/worker_layer_percentage_without_communication_wait.png')

    # Layer Percentage Without Communication Wait and IO
    worker_layer_offline_percentage_df = worker_stats_df[worker_stats_df['step_id'] == worker_stats_df['step_id'].max()].filter(items=['batch_size', 'rank', 'local_rank', 'offline_bottom_mlp_percentage', 'offline_embedding_percentage', 'offline_communication_percentage', 'offline_interaction_percentage', 'offline_top_mlp_percentage']).sort_values('batch_size').reset_index(drop=True)
    worker_layer_offline_percentage_df.to_csv(f'{output}/offline_layer_percentage.csv', header=['Batch Size', 'Rank', 'Local Rank', 'Bottom MLP(%)', 'Emdedding(%)', 'Communication(%)', 'Interaction(%)', 'Top MLP(%)'], index=False)
    plots(df=worker_layer_offline_percentage_df,
        key = 'batch_size',
        kind = 'stacked_bar',
        x = 'rank',
        y = ['offline_bottom_mlp_percentage', 'offline_embedding_percentage', 'offline_communication_percentage', 'offline_interaction_percentage', 'offline_top_mlp_percentage'],
        title = 'Worker Layer Percentage Without Communication Wait and IO',
        ylabel = 'Latency(ns)',
        legend=['Bottom MLP(%)', 'Embedding(%)', 'Communication(%)', 'Interaction(%)', 'Top MLP(%)'],
        logx = False,
        logy = False).savefig(f'{output}/worker_layer_percentage_without_communication_wait_and_io.png')

    return 0

exit(main())