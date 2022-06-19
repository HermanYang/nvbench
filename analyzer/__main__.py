import argparse
import sqlite3
import pathlib
from turtle import color
from matplotlib.colorbar import Colorbar
import pandas 
import matplotlib.pyplot as plt
import numpy

parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, required=True, help='Path to nvbench results')

def analyze(nsysdb_path, power_draw_path):
    worker_number = int(power_draw_path.split('_')[1])
    batch_size = int(power_draw_path.split('_')[2])
    if worker_number > batch_size:
        worker_number = batch_size

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

    query_category_map = ''' 
        SELECT category, text
        FROM NVTX_EVENTS 
        WHERE nvtx_event_map(eventType) == "NvtxCategory" AND category IS NOT NULL GROUP BY category, text;
    '''

    query_string_map = ''' 
        SELECT id, value
        FROM StringIds
    '''

    query_e2e_step_stats = '''
        SELECT NVTX_EVENTS.globalTid, NVTX_EVENTS.start, NVTX_EVENTS.end, NVTX_EVENTS.text
        FROM NVTX_EVENTS 
        WHERE nvtx_event_map(eventType) == "NvtxPushPopRange" AND NVTX_EVENTS.text LIKE "step%";
    '''

    con.create_function('nvtx_event_map', 1, lambda x: nvtx_event_map[x])

    category_map = dict(con.execute(query_category_map).fetchall())
    con.create_function('category_map', 1, lambda x: category_map[x])

    con.create_function('step_map', 1, lambda x: int(x.split(' ')[1]))

    string_map = dict(con.execute(query_string_map).fetchall())
    con.create_function('string_map', 1, lambda x: string_map[x])

    worker_power_draw = []
    power_draw_df = pandas.DataFrame(pandas.read_csv(power_draw_path))
    worker_power_draw = power_draw_df.max().to_list()
    worker_power_draw.sort(reverse=True)
    worker_power_draw = worker_power_draw[:worker_number]

    global_tid_list = pandas.read_sql_query(query_e2e_step_stats, con)['globalTid'].unique()
    global_tid_map = dict()
    for i, global_tid in enumerate(global_tid_list):
        global_tid_map[global_tid] = i
    con.create_function('global_tid_map', 1, lambda x: global_tid_map[x])

    con.executescript(create_stats_views_sql)

    input_stats_df = pandas.read_sql_query('SELECT * FROM input_stats', con)
    communication_stats_df = pandas.read_sql_query('SELECT * FROM communication_stats', con)
    output_stats_df = pandas.read_sql_query('SELECT * FROM output_stats', con)
    bottom_mlp_stats_df = pandas.read_sql_query('SELECT * FROM bottom_mlp_stats', con)
    embedding_stats_df = pandas.read_sql_query('SELECT * FROM embedding_stats', con)
    interaction_stats_df = pandas.read_sql_query('SELECT * FROM interaction_stats', con)
    top_mlp_stats_df = pandas.read_sql_query('SELECT * FROM top_mlp_stats', con)
    inference_stats_df = pandas.read_sql_query('SELECT * FROM inference_stats', con)

    cur.close()

    # merge all stats
    worker_stats_df = inference_stats_df.merge(
        input_stats_df.rename(columns={'host_duration':'input_host_duration', 'device_duration':'input_device_duration', 'bytes':'input_bytes', 'throughput':'input_throughput'}), 
        how='left', on=['card_id','globalTid','step_id'])
    worker_stats_df = worker_stats_df.merge(communication_stats_df.rename(columns={'host_duration':'communication_host_duration', 'device_duration':'communication_device_duration'}), how='left', on=['card_id','globalTid','step_id'])
    worker_stats_df = worker_stats_df.merge(bottom_mlp_stats_df.rename(columns={'host_duration':'bottom_mlp_duration', 'device_duration':'bottom_mlp_device_duration'}), how='left',  on=['card_id','globalTid','step_id'])
    worker_stats_df = worker_stats_df.merge(embedding_stats_df.rename(columns={'host_duration':'embedding_duration', 'device_duration':'embedding_device_duration'}), how='left', on=['card_id','globalTid','step_id'])
    worker_stats_df = worker_stats_df.merge(interaction_stats_df.rename(columns={'host_duration':'interaction_host_duration', 'device_duration':'interaction_device_duration'} ), how='left', on=['card_id','globalTid','step_id'])
    worker_stats_df = worker_stats_df.merge(top_mlp_stats_df.rename(columns={'host_duration':'top_mlp_host_duration', 'device_duration':'top_mlp_device_duration'}), how='left', suffixes=['', '_top_mlp'], on=['card_id','globalTid','step_id'])
    worker_stats_df = worker_stats_df.merge(output_stats_df.rename(columns={'host_duration':'output_host_duration', 'device_duration':'output_device_duration', 'bytes':'output_bytes', 'throughput':'output_throughput'}), how='left', on=['card_id','globalTid','step_id'])
    worker_stats_df = worker_stats_df.fillna(0)

    # estimate ideal_e2e_duration, only consider input + compute + communication + output time on device
    worker_stats_df = worker_stats_df.assign(ideal_e2e_duration = lambda stats: stats['input_device_duration'] + stats['bottom_mlp_device_duration'] + stats['embedding_device_duration'] + stats['communication_device_duration'] + stats['interaction_device_duration'] + stats['top_mlp_device_duration'] + stats['output_device_duration'])
    worker_stats_df = worker_stats_df.assign(ideal_e2e_duration_without_io = lambda stats: stats['bottom_mlp_device_duration'] + stats['embedding_device_duration'] + stats['communication_device_duration'] + stats['interaction_device_duration'] + stats['top_mlp_device_duration'])

    # layers ratio
    worker_stats_df = worker_stats_df.assign(bottom_mlp_ratio = lambda stats: stats['bottom_mlp_device_duration'] / stats['ideal_e2e_duration_without_io'])
    worker_stats_df = worker_stats_df.assign(embedding_ratio = lambda stats: stats['embedding_device_duration'] / stats['ideal_e2e_duration_without_io'])
    worker_stats_df = worker_stats_df.assign(communication_ratio = lambda stats: stats['communication_device_duration'] / stats['ideal_e2e_duration_without_io'])
    worker_stats_df = worker_stats_df.assign(interaction_ratio = lambda stats: stats['interaction_device_duration'] / stats['ideal_e2e_duration_without_io'])
    worker_stats_df = worker_stats_df.assign(top_mlp_ratio = lambda stats: stats['top_mlp_device_duration'] / stats['ideal_e2e_duration_without_io'])


    system_stats_df = worker_stats_df[['step_id', 'host_duration', 'ideal_e2e_duration', 'ideal_e2e_duration_without_io']]
    system_stats_df = system_stats_df.groupby(['step_id']).max()

    nanosecond_to_second = 1e-9
    system_stats_df = system_stats_df.assign(real_ips = lambda stats: batch_size / (stats['host_duration'] * nanosecond_to_second))
    system_stats_df = system_stats_df.assign(ideal_ips = lambda stats: batch_size / (stats['ideal_e2e_duration'] * nanosecond_to_second))
    system_stats_df = system_stats_df.assign(ideal_ips_without_io = lambda stats: batch_size / (stats['ideal_e2e_duration_without_io'] * nanosecond_to_second))

    # power draw estimation with real ips
    system_stats_df = system_stats_df.assign(ipw = lambda stats: stats['real_ips'] / sum(worker_power_draw))

    return worker_stats_df, system_stats_df


def main():
    args = parser.parse_args()
    path = pathlib.Path(args.path)

    if not path.exists():
        raise ValueError('Path does not exist: {}'.format(path))

    ips_ipw_records = list()
    layer_ratio_records = list()

    for file in path.glob('**/*.sqlite'):
        nsysdb_path = f'{path}/{file.stem}.sqlite'
        power_draw_path = f'{path}/{file.stem}_power_draw.csv'

        if not pathlib.Path(nsysdb_path).exists():
            raise ValueError('Path does not exist: {}'.format(nsysdb_path))

        if not pathlib.Path(power_draw_path).exists():
            raise ValueError('Path does not exist: {}'.format(power_draw_path))

        batch_size = int(file.stem.split('_')[-1])
        worker_number = int(file.stem.split('_')[-2])
        worker_number = min(batch_size, worker_number)
        worker_stats_df, system_stats_df = analyze(nsysdb_path, power_draw_path)
        system_stats_df.to_csv(f'{path}/{file.stem}_system_stats.csv')
        worker_stats_df = worker_stats_df[worker_stats_df['step_id'] == 9]
        ips_ipw_records.append({'batch_size': batch_size, 'real_ips': system_stats_df['real_ips'].values[-1],  'ideal_ips': system_stats_df['ideal_ips'].values[-1], 'ideal_ips_without_io': system_stats_df['ideal_ips_without_io'].values[-1], 'ipw': system_stats_df['ipw'].values[-1]})

        for card_id in range(worker_number):
            layer_ratio_records.append({'batch_size': batch_size, 'card_id': card_id, 'bottom_mlp_ratio': worker_stats_df['bottom_mlp_ratio'].values[card_id], 'embedding_ratio': worker_stats_df['embedding_ratio'].values[card_id], 'communication_ratio': worker_stats_df['communication_ratio'].values[card_id], 'interaction_ratio': worker_stats_df['interaction_ratio'].values[card_id], 'top_mlp_ratio': worker_stats_df['top_mlp_ratio'].values[card_id]})

    ips_ipw_df = pandas.DataFrame.from_records(ips_ipw_records).sort_values('batch_size')
    ips_ipw_df.to_csv(f'{path}/ips_ipw.csv', index=False)

    layer_ratio_df = pandas.DataFrame.from_records(layer_ratio_records).sort_values('batch_size')
    layer_ratio_df.to_csv(f'{path}/layer_ratio.csv', index=False)

    # generate plots
    plt.clf()
    xtick_labels = ips_ipw_df['batch_size'].to_numpy().astype(str)
    ips_ipw_df['batch_size'] = ips_ipw_df['batch_size'].transform(numpy.log2)
    axs = ips_ipw_df.plot(x='batch_size', y=['real_ips' ],
        kind='line', grid=True, marker = 'o', title='Throughput', xlabel='Batch Size', ylabel='Inference Per Second', logy = True)
    axs.set_xticks(ips_ipw_df['batch_size'].to_numpy())
    axs.set_xticklabels(labels = xtick_labels)
    plt.legend(['IPS'])
    plt.tight_layout()
    plt.savefig(f'{path}/ips.png')

    plt.clf()
    axs = ips_ipw_df.plot(x='batch_size', y=['ipw'], kind='line', grid=True, marker = 'o', title='Energy Efficiency', xlabel='Batch Size', ylabel='Inference Per Watt')
    axs.set_xticks(ips_ipw_df['batch_size'].to_numpy())
    axs.set_xticklabels(labels = xtick_labels)
    plt.legend(['IPW'])
    plt.tight_layout()
    plt.savefig(f'{path}/ipw.png')

    plt.clf()
    plt.ylim(0, 1)
    scale = 6
    ncol = 3
    batch_size_list = layer_ratio_df['batch_size'].unique()
    nrow = numpy.ceil(batch_size_list.size / ncol).astype(int)
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(nrow * scale, ncol * scale))
    axes[-1][-1].axis('off')
    axes[-1][-2].axis('off')
    for i, batch_size in enumerate(batch_size_list):
        batch_layer_ratio_df = layer_ratio_df[layer_ratio_df['batch_size'] == batch_size].filter(items=['card_id', 'bottom_mlp_ratio', 'embedding_ratio', 'communication_ratio', 'interaction_ratio', 'top_mlp_ratio'])
        batch_layer_ratio_df.sort_values('card_id', inplace=True)
        axs = batch_layer_ratio_df.plot(ax = axes[i // ncol][i % ncol], x='card_id', y=['bottom_mlp_ratio', 'embedding_ratio', 'communication_ratio', 'interaction_ratio', 'top_mlp_ratio'],  kind='bar', grid=True, title=f'Batch {batch_size}', xlabel='Worker', ylabel='Ratio', stacked = True)
        axs.legend(['Bottom MLP', 'Embedding', 'Communication', 'Interaction','Top MLP'], loc = 'best',  framealpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{path}/layer_ratio.png')
    return 0

exit(main())