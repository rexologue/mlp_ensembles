import os

from trainer import Trainer
from utils.common_functions import read_file

classes = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

path_to_cfgs = '/home/duka/job/mlp/configs'

if __name__ == '__main__':
    for cls in classes:
        cls_cfg = os.path.join(path_to_cfgs, cls)

        cfg_list = [os.path.join(cls_cfg, x) for x in os.listdir(cls_cfg)]

        for cfg_path in cfg_list:
            cfg = read_file(cfg_path)

            expert_name = cls + '-' + cfg_path.split('_')[-1].split('.')[0]
            print(f"Train {expert_name} expert")

            trainer = Trainer(cfg, True)
            trainer.fit()
    