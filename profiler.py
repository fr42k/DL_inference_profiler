#!/usr/bin/python3.6
__author__ = 'Yunjie Wang'
__email__ = 'fr42k.w@gmail.com'
import argparse
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm

if __name__  == '__main__':
    unit_info = 'unit for size: byte\nunit for time: second'
    print(unit_info)
    parser = argparse.ArgumentParser(description=
             'Keras Profiler for (type:) [1] VGG 19 (default), [2] MobileNet V1, [3] ResNet 50, [4] Inception V3')
    parser.add_argument("--type", dest="type", help="type of above nets, default: 1", default=1, type=int)
    parser.add_argument("--bandwidth", dest="bandwidth", default=10*1024*1024*1024, type=float,
                        help="External – internal memory bandwidth in bytes / second, default: 10 GB/sec")
    parser.add_argument("--mem", dest="mem", default=16*1024*1024, type=float,
                        help="Internal memory size in bytes, default: 16 MB")
    parser.add_argument("--M", dest="M", default=32, type=int,
                        help="Matrix multiplication primitive size M in BLAS notation, default: 32")
    parser.add_argument("--N", dest="N", default=32, type=int,
                        help="Matrix multiplication primitive size N in BLAS notation, default: 32")
    parser.add_argument("--K", dest="K", default=32, type=int,
                        help="Matrix multiplication primitive size K in BLAS notation, default: 32")
    parser.add_argument("--mparal", dest="mparal", default=1, type=int,
                        help="Number of matrix multiplication primitives operating in parallel, default: 1")
    parser.add_argument("--mspeed", dest="mspeed", default=1e9/32, type=float,
                        help="Number of matrix multiplication primitive completions per second, default: 1e9/32")
    parser.add_argument("--vn", dest="vn", default=32, type=int,
                        help="Vector primitive size N x 1, default: 32")
    parser.add_argument("--vparal", dest="vparal", default=1, type=int,
                        help="Number of vector multiplication primitives operating in parallel, default: 1")
    parser.add_argument("--vspeed", dest="vspeed", default=1e9, type=float,
                        help="Number of vector primitive completions per second, default: 1e9")
    parser.add_argument("--outfile", dest="outfile", default="_profiler_result", type=str,
                        help="suffix of output filename, default: _profiler_result")

    args = parser.parse_args()

    from keras.applications.resnet50 import ResNet50
    from keras.applications.vgg19 import VGG19
    from keras.applications.inception_v3 import InceptionV3
    from keras.applications.mobilenet import MobileNet

    base_model_dict = {
        1: VGG19(weights='imagenet'),
        2: MobileNet(weights='imagenet'),
        3: ResNet50(weights='imagenet'),
        4: InceptionV3(weights='imagenet', input_shape=(224, 224, 3))
    }

    base_file_name = {
        1: "VGG19", 2: "MobileNetV1", 3: "ResNet50", 4: "InceptionV3"
    }

    Free_Layer = ['ReLU', 'InputLayer', 'ZeroPadding2D', 'Flatten', 'Concatenate', 'Dropout']
    Free_Activation = ['relu', 'linear']
    base_model = base_model_dict[args.type]

    # used internal memory
    mem = 0
    # {detail for each layer}
    stat_layer = []
    stat_total = {'name': "TOTAL",
        'input_fm_move_time': 0, 'filter_coefficient_move_time': 0, 'output_fm_move_time': 0, 'total_data_move_time': 0,
        'matrix_compute_time': 0, 'vector_compute_time': 0, "total_compute_time": 0,
        'serial_total_move_compute_time': 0, 'parallel_total_move_compute_time': 0}

    def get_bytes_matrix(shape):
        return shape[1] * shape[2] * shape[3]

    def move_func(MAC, args):
        return MAC * 1.0 / args.bandwidth

    def comp_matrix_time(MAC, args):
        return MAC * 1.0 / (args.M * args.N * args.K * args.mspeed * args.mparal)

    def comp_vec_time(MAC1, MAC2, args):
        return MAC1 * MAC2 * 1.0 / (args.vn * args.vspeed * args.vparal)

    def comp_default_time(MAC1, MAC2, args):
        return np.maximum(MAC1 * 1.0, MAC2 * 1.0) / (args.vn * args.vspeed * args.vparal)

    for layer in tqdm(base_model.layers):
        conf = layer.get_config()
        type = layer.__class__.__name__
        stat_here = {"name": layer.name, "type": type}

        # per layer part
        # input
        def fill_stat_input(shape):
            if len(shape) == 4:
                stat_here['#input_fm'] += shape[3]
                stat_here['#input_fm_rows'] += shape[1]
                stat_here['#input_fm_cols'] += shape[2]
                stat_here['#input_fm_bytes'] += get_bytes_matrix(shape)
                stat_here['input_fm_up_sampling_ratio'] = (1, 1)

            else:
                stat_here['#input_fm'] += 1
                stat_here['#input_fm_rows'] += shape[1]
                stat_here['#input_fm_cols'] += 1
                stat_here['#input_fm_bytes'] += shape[1]
                stat_here['input_fm_up_sampling_ratio'] = 1

        stat_here['#input_fm'] = 0
        stat_here['#input_fm_rows'] = 0
        stat_here['#input_fm_cols'] = 0
        stat_here['#input_fm_bytes'] = 0
        if type in ['Add', 'Concatenate']:
            for shape in layer.input_shape:
                fill_stat_input(shape)
        else:
            fill_stat_input(layer.input_shape)

        mem += stat_here['#input_fm_bytes']
        if mem > args.mem:
            mem -= stat_here['#input_fm_bytes']
            stat_here["input_fm_location"] = "external"
        else:
            stat_here["input_fm_location"] = "internal"

        # output
        if len(layer.output_shape) == 4:
            stat_here['#output_fm'] = layer.output_shape[3]
            stat_here['#output_fm_rows'] = layer.output_shape[1]
            stat_here['#output_fm_cols'] = layer.output_shape[2]
            stat_here['#output_fm_bytes'] = get_bytes_matrix(layer.output_shape)
            stat_here['output_fm_down_sampling_ratio'] = conf['strides'] if 'strides' in conf else 'NA'
        else:
            stat_here['#output_fm'] = 1
            stat_here['#output_fm_rows'] = layer.output_shape[1]
            stat_here['#output_fm_cols'] = 1
            stat_here['#output_fm_bytes'] = layer.output_shape[1]
            stat_here['output_fm_down_sampling_ratio'] = layer.output_shape[1] * 1.0 / layer.input_shape[1] \
            if len(layer.input_shape) == 2 else 'NA'

        mem += stat_here['#output_fm_bytes']
        if mem > args.mem:
            mem -= stat_here['#output_fm_bytes']
            stat_here["output_fm_location"] = "external"
        else:
            stat_here["output_fm_location"] = "internal"

        # feature map
        stat_here['filter_grouping'] = 1 if len(layer.output_shape) == 4 else 0
        if type == 'DepthwiseConv2D':
            stat_here['filter_grouping'] = layer.input_shape[3]
        stat_here['filter_up_sampling_ratio'] = conf['dilation_rate'] if 'dilation_rate' in conf else 'NA'
        if 'kernel_size' in conf:
            stat_here['#filter_rows'], stat_here['#filter_cols'] = conf['kernel_size']
        elif 'pool_size' in conf:
            stat_here['#filter_rows'], stat_here['#filter_cols'] = conf['pool_size']
            stat_here['filter_up_sampling_ratio'] = 1
        else:
            stat_here['#filter_rows'], stat_here['#filter_cols'] = 0, 0
        stat_here['filter_coefficient_location'] = 'external'

        stat_here['#filter_bytes'] = stat_here['#input_fm'] * stat_here['#output_fm'] * stat_here['#filter_rows'] * \
                                        stat_here['#filter_cols']
        if type == 'DepthwiseConv2D':
            stat_here['#filter_bytes'] = stat_here['filter_grouping'] * stat_here['#filter_rows'] * \
                                         stat_here['#filter_cols']

        # time for data movement
        stat_here['input_fm_move_time'] = move_func(stat_here['#input_fm_bytes'], args)
        stat_here['output_fm_move_time'] = move_func(stat_here['#output_fm_bytes'], args)
        stat_here['filter_coefficient_move_time'] = move_func(stat_here['#filter_bytes'], args)
        stat_here['total_data_move_time'] = stat_here['input_fm_move_time'] + stat_here['output_fm_move_time'] + \
                                            stat_here['filter_coefficient_move_time']
        # time for computation
        stat_here['matrix_compute_time'] = 0
        stat_here['vector_compute_time'] = 0
        if type == 'Conv2D':
            stat_here['matrix_compute_time'] += comp_matrix_time(stat_here['#filter_rows'] * stat_here['#filter_cols'] *\
                stat_here['#input_fm'] * stat_here['#output_fm'] *\
                stat_here['#output_fm_rows'] * stat_here['#output_fm_cols'], args)
            if 'activation' in conf and conf['activation'] not in Free_Activation:
                stat_here['vector_compute_time'] += comp_default_time(stat_here['#input_fm'] *\
                    stat_here['#input_fm_rows'] * stat_here['#input_fm_cols'], stat_here['#output_fm'] *\
                    stat_here['#output_fm_rows'] * stat_here['#output_fm_cols'], args)

        elif type == 'DepthwiseConv2D':
            stat_here['matrix_compute_time'] += comp_matrix_time(stat_here['#filter_rows'] * stat_here['#filter_cols'] *\
                stat_here['#input_fm'] *\
                stat_here['#output_fm_rows'] * stat_here['#output_fm_cols'], args)
            if 'activation' in conf and conf['activation'] not in Free_Activation:
                stat_here['vector_compute_time'] += comp_default_time(stat_here['#input_fm'] *\
                    stat_here['#input_fm_rows'] * stat_here['#input_fm_cols'], stat_here['#output_fm'] *\
                    stat_here['#output_fm_rows'] * stat_here['#output_fm_cols'], args)

        elif type in Free_Layer:
            pass
        elif type.find('Pooling') != -1 or type == 'Dense':
            stat_here['vector_compute_time'] += comp_vec_time(stat_here['#input_fm'] * \
                              stat_here['#input_fm_rows'] * stat_here['#input_fm_cols'], stat_here['#output_fm'] * \
                              stat_here['#output_fm_rows'] * stat_here['#output_fm_cols'], args)
        else :
            stat_here['vector_compute_time'] += comp_default_time(stat_here['#input_fm'] * \
                              stat_here['#input_fm_rows'] * stat_here['#input_fm_cols'], stat_here['#output_fm'] * \
                              stat_here['#output_fm_rows'] * stat_here['#output_fm_cols'], args)

        stat_here['total_compute_time'] = stat_here['matrix_compute_time'] + stat_here['vector_compute_time']

        # push back layer info
        stat_layer.append(stat_here)

        # total time for this layer
        stat_here['serial_total_move_compute_time'] = stat_here['total_data_move_time'] + \
                                                      stat_here['total_compute_time']
        stat_here['parallel_total_move_compute_time'] = np.maximum(stat_here['total_data_move_time'],
                                                      stat_here['total_compute_time'])

        # total part
        stat_total['input_fm_move_time'] += stat_here['input_fm_move_time']
        stat_total['filter_coefficient_move_time'] += stat_here['filter_coefficient_move_time']
        stat_total['output_fm_move_time'] += stat_here['output_fm_move_time']
        stat_total['total_data_move_time'] += stat_here['total_data_move_time']

        stat_total['matrix_compute_time'] += stat_here['matrix_compute_time']
        stat_total['vector_compute_time'] += stat_here['vector_compute_time']
        stat_total['total_compute_time'] += stat_here['total_compute_time']

        stat_total['serial_total_move_compute_time'] += stat_here['serial_total_move_compute_time']
        stat_total['parallel_total_move_compute_time'] += stat_here['parallel_total_move_compute_time']

    print('-'*50)
    print(unit_info)
    print(args)
    print(stat_total)
    print('-'*50)

    filename = base_file_name[args.type] + args.outfile
    with open("%s.csv" % filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=stat_layer[0].keys())
        writer.writeheader()
        for stat in stat_layer:
            writer.writerow(stat)
        writer.writerow({})
        writer.writerow(stat_total)
        print("Successfully written %s.csv" % filename)

    df = pd.read_csv("%s.csv" % filename)
    df.to_html("%s.html" % filename, na_rep='')
    print("Successfully written %s.html" % filename)
