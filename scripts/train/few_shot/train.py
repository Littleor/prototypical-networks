import json
import os
from functools import partial

import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchnet as tnt

import protonets.utils.data as data_utils
import protonets.utils.log as log_utils
import protonets.utils.model as model_utils
from protonets.engine import Engine


def main(opt):
    # log目录不存在就创建
    if not os.path.isdir(opt['log.exp_dir']):
        os.makedirs(opt['log.exp_dir'])

    # 把参数保存到opt.json中
    with open(os.path.join(opt['log.exp_dir'], 'opt.json'), 'w') as f:
        json.dump(opt, f)
        f.write('\n')
    # 追踪每轮训练中的情况
    trace_file = os.path.join(opt['log.exp_dir'], 'trace.txt')

    # Postprocess arguments
    opt['model.x_dim'] = list(map(int, opt['model.x_dim'].split(',')))
    opt['log.fields'] = opt['log.fields'].split(',')

    # 初始化种子 保证网络初始值一致
    torch.manual_seed(1234)

    # 显卡训练
    if opt['data.cuda']:
        torch.cuda.manual_seed(1234)

    # 根据需求加载数据集
    if opt['data.trainval']:
        data = data_utils.load(opt, ['trainval'])
        train_loader = data['trainval']
        val_loader = None
    else:
        data = data_utils.load(opt, ['train', 'val'])
        train_loader = data['train']
        val_loader = data['val']

    # 加载模型
    model = model_utils.load(opt)

    if opt['data.cuda']:
        model.cuda()

    # tnt.Engine -> 方便训练和评估模型
    engine = Engine()

    meters = {'train': {field: tnt.meter.AverageValueMeter() for field in opt['log.fields']}}

    if val_loader is not None:
        meters['val'] = {field: tnt.meter.AverageValueMeter() for field in opt['log.fields']}

    def on_start(state):
        if os.path.isfile(trace_file):
            os.remove(trace_file)
        state['scheduler'] = lr_scheduler.StepLR(state['optimizer'], opt['train.decay_every'], gamma=0.5)

    engine.hooks['on_start'] = on_start

    def on_start_epoch(state):
        for split, split_meters in meters.items():
            for field, meter in split_meters.items():
                meter.reset()
        state['scheduler'].step()

    engine.hooks['on_start_epoch'] = on_start_epoch

    def on_update(state):
        for field, meter in meters['train'].items():
            meter.add(state['output'][field])

    engine.hooks['on_update'] = on_update

    def on_end_epoch(hook_state, state):
        if val_loader is not None:
            if 'best_loss' not in hook_state:
                hook_state['best_loss'] = np.inf
            if 'wait' not in hook_state:
                hook_state['wait'] = 0

        if val_loader is not None:
            model_utils.evaluate(state['model'],
                                 val_loader,
                                 meters['val'],
                                 desc="Epoch {:d} valid".format(state['epoch']))

        meter_vals = log_utils.extract_meter_values(meters)
        print("Epoch {:02d}: {:s}".format(state['epoch'], log_utils.render_meter_values(meter_vals)))
        meter_vals['epoch'] = state['epoch']
        with open(trace_file, 'a') as f:
            json.dump(meter_vals, f)
            f.write('\n')

        if val_loader is not None:
            if meter_vals['val']['loss'] < hook_state['best_loss']:
                hook_state['best_loss'] = meter_vals['val']['loss']
                print("==> best model (loss = {:0.6f}), saving model...".format(hook_state['best_loss']))

                state['model'].cpu()
                torch.save(state['model'], os.path.join(opt['log.exp_dir'], 'best_model.pt'))
                if opt['data.cuda']:
                    state['model'].cuda()

                hook_state['wait'] = 0
            else:
                hook_state['wait'] += 1

                if hook_state['wait'] > opt['train.patience']:
                    print("==> patience {:d} exceeded".format(opt['train.patience']))
                    state['stop'] = True
        else:
            state['model'].cpu()
            torch.save(state['model'], os.path.join(opt['log.exp_dir'], 'best_model.pt'))
            if opt['data.cuda']:
                state['model'].cuda()

    engine.hooks['on_end_epoch'] = partial(on_end_epoch, {})

    engine.train(
        model=model,
        loader=train_loader,
        optim_method=getattr(optim, opt['train.optim_method']),
        optim_config={'lr': opt['train.learning_rate'],
                      'weight_decay': opt['train.weight_decay']},
        max_epoch=opt['train.epochs']
    )
