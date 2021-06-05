from tqdm import tqdm


# import torchnet.engine.engine

# 手动写了个 tnt.Engine 具体可见上面注释中的
class Engine(object):
    def __init__(self):
        hook_names = ['on_start', 'on_start_epoch', 'on_sample', 'on_forward',
                      'on_backward', 'on_end_epoch', 'on_update', 'on_end']

        self.hooks = {}
        for hook_name in hook_names:
            self.hooks[hook_name] = lambda state: None

    # 训练方法
    def train(self, **kwargs):
        state = {
            'model': kwargs['model'],
            'loader': kwargs['loader'],
            'optim_method': kwargs['optim_method'],
            'optim_config': kwargs['optim_config'],
            'max_epoch': kwargs['max_epoch'],
            'epoch': 0,  # epochs done so far
            't': 0,  # samples seen so far
            'batch': 0,  # samples seen in current epoch
            'stop': False
        }
        # 设置优化器
        state['optimizer'] = state['optim_method'](state['model'].parameters(), **state['optim_config'])

        # 调用hook
        self.hooks['on_start'](state)
        # 训练过程中
        while state['epoch'] < state['max_epoch'] and not state['stop']:
            # 设置为训练状态：Sets the module in training mode.
            state['model'].train()
            # hook
            self.hooks['on_start_epoch'](state)
            # 获取批次大小
            state['epoch_size'] = len(state['loader'])

            for sample in tqdm(state['loader'], desc="Epoch {:d} train".format(state['epoch'] + 1)):
                state['sample'] = sample
                self.hooks['on_sample'](state)
                # 清空梯度
                state['optimizer'].zero_grad()
                # 计算损失
                loss, state['output'] = state['model'].loss(state['sample'])
                self.hooks['on_forward'](state)
                # 计算梯度
                loss.backward()
                self.hooks['on_backward'](state)
                # 优化器
                state['optimizer'].step()

                state['t'] += 1
                state['batch'] += 1
                self.hooks['on_update'](state)

            state['epoch'] += 1
            state['batch'] = 0
            self.hooks['on_end_epoch'](state)

        self.hooks['on_end'](state)
