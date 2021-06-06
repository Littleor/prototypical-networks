import json
import os
import subprocess

from protonets.utils import format_opts, merge_dict
from protonets.utils.log import load_trace


# 验证模型
def main(opt):
    # 去读模型
    result_dir = os.path.dirname(opt['model.model_path'])

    # get target training loss to exceed
    trace_file = os.path.join(result_dir, 'trace.txt')
    trace_vals = load_trace(trace_file)
    best_epoch = trace_vals['val']['loss'].argmin()

    # load opts
    # 加载配置
    model_opt_file = os.path.join(os.path.dirname(opt['model.model_path']), 'opt.json')
    with open(model_opt_file, 'r') as f:
        model_opt = json.load(f)

    # override previous training ops
    # 改写部分配置
    model_opt = merge_dict(model_opt, {
        'log.exp_dir': os.path.join(model_opt['log.exp_dir'], 'trainval'),
        'data.trainval': True,
        'train.epochs': best_epoch + model_opt['train.patience'],
    })
    # 调用bash指令运行并获取返回值
    subprocess.call(
        ['python', os.path.join(os.getcwd(), 'scripts/train/few_shot/run_train.py')] + format_opts(model_opt))
    # 其实这里还是去调用了run_train，只是手动输入了参数而已。
