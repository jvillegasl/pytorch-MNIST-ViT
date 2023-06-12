import argparse
import collections
import os

import torch

import model.model as module_arch
from parse_config import ConfigParser


def main(config: ConfigParser):
    # logger = config.get_logger('export')

    model = config.init_obj('arch', module_arch)

    # logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    model = model.to('cpu')
    model.eval()

    folder_path = './exports'
    os.makedirs(folder_path, exist_ok=True)

    file_name = config['export']['file_name'] + '.onnx'

    file_path = os.path.join(folder_path, file_name)

    x = torch.randn(*config['export']['dummy_input'],
                    requires_grad=True).to('cpu')
    y = model(x)

    torch.onnx.export(
        model,
        x,
        file_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=True
    )


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['-f', '--file_name'], type=str, target='export;file_name'),
    ]

    config = ConfigParser.from_args(args, options)
    main(config)
