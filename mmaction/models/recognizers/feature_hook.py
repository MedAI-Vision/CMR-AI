import torch
import numpy as np


class HookTool:
    def __init__(self, name):
        self.idx = 0
        self.name = name

    def hook_fun1(self, module, in_fea, out_fea):
        self.idx += 1
        data1 = in_fea[0].cpu().numpy().squeeze()
        data1 = data1.reshape(data1.shape[0], -1)
        np.save('/media/data/yanran/CKA/' + self.name + '_in_' + str(self.idx) + '.npy', data1)
        data2 = out_fea.cpu().numpy().squeeze()
        data2 = data2.reshape(data2.shape[0], -1)
        np.save('/media/data/yanran/CKA/'+self.name+'_out_'+str(self.idx)+'.npy', data2)

    def hook_fun2(self, module, in_fea, out_fea):
        self.idx += 1
        data = in_fea[0].cpu().numpy().squeeze()
        data = data.reshape(data.shape[0], -1)
        np.save('/media/data/yanran/CKA/' + self.name + '_in_' + str(self.idx) + '.npy', data)
    
    def hook_fun3(self, module, in_fea, out_fea):
        self.idx += 1
        data = out_fea.cpu().numpy().squeeze()
        data = data.reshape(data.shape[0], -1)
        np.save('/media/data/yanran/CKA/' + self.name + '_out_' + str(self.idx) + '.npy', data)


def get_fea_by_hook(model):
    for n, m in model.named_modules():
        if 'downsample' not in n and len(n.split('.')) is 4:
            if 'blocks.0' in n:
                cur_hook = HookTool(n)
                m.register_forward_hook(cur_hook.hook_fun1)
            else:
                cur_hook = HookTool(n)
                m.register_forward_hook(cur_hook.hook_fun3)
        elif 'norm2' in n:
            cur_hook = HookTool(n)
            m.register_forward_hook(cur_hook.hook_fun2)






