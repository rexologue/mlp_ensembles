import os
import copy
import numpy as np

from config import exp_cfg
from utils.common_functions import write_file
from utils.enums import WeightsInitType, LayerType, TransformType


NUM_CFGS = 10
CLASSES = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']


LAYERS_AMOUNT = [2, 3, 4]
RESOLUTION = [32, 48]  # 32 ... 48

TRANSFORM_TYPE = [
    {'type': TransformType.Standardize, 'params': {'mean': 0.50708866, 'std': 0.25496453}},
    {'type': TransformType.Normalize, 'params': {'a': -1, 'b': 1}}
]

INIT_TYPE = [WeightsInitType.he, WeightsInitType.xavier, WeightsInitType.xavier_normalized]

LR = [0.005, 0.01]
W_DECAY = [0.001, 0.01]


def generate_mlp_architecture(
    num_layers: int,
    input_features: int,
    output_features: int,
    min_features: int = 16
) -> list:
    """
    Генерирует архитектуру MLP с заданными параметрами.
    
    Args:
        num_layers: Количество линейных слоев
        input_features: Количество входных признаков
        output_features: Количество выходных признаков
        min_features: Минимальное количество признаков в скрытых слоях
        
    Returns:
        Список слоев с параметрами
    """
    if num_layers < 1:
        raise ValueError("Количество слоев должно быть не менее 1")
    if input_features < output_features:
        raise ValueError("Количество входных признаков должно быть >= выходных")
    
    layers = []
    current_features = input_features
    
    # Базовый шаг уменьшения признаков с небольшим случайным отклонением
    base_step = (input_features - output_features) / num_layers
    steps = [int(base_step * (0.8 + 0.4 * np.random.random())) for _ in range(num_layers-1)]
    steps.append(input_features - output_features - sum(steps))  # Последний шаг
    
    for i in range(num_layers):
        # Линейный слой
        next_features = current_features - steps[i]
        
        # Гарантируем, что количество признаков не станет слишком маленьким
        if i != num_layers - 1:
            next_features = max(next_features, min_features)
            next_features = min(next_features, current_features - 1)
        
        layers.append({
            'type': LayerType.Linear,
            'params': {
                'in_features': current_features,
                'out_features': next_features
            }
        })
        
        # Добавляем LeakyReLU и Dropout, кроме последнего слоя
        if i != num_layers - 1:
            layers.append({
                'type': LayerType.LeakyReLU,
                'params': {'alpha': round(0.05 + 0.15 * np.random.random(), 2)}
            })
            
            # Dropout с вероятностью от 0.1 до 0.5
            dropout_p = np.random.uniform(0.05, 0.35)
            layers.append({
                'type': LayerType.Dropout,
                'params': {'p': round(dropout_p, 2)}
            })
        
        current_features = next_features
    
    return layers


def generate_config(save_path: str, cls: str):
    # Генерация случайных параметров
    layers = np.random.permutation(
        np.linspace(start=LAYERS_AMOUNT[0], stop=LAYERS_AMOUNT[-1], num=NUM_CFGS, dtype=int)
    ).tolist()
    
    resolutions = [{'type': TransformType.Resize, 'params': {'size': (i, i)}} for i in np.random.permutation(
        np.linspace(start=RESOLUTION[0], stop=RESOLUTION[-1], num=NUM_CFGS, dtype=int)
    )]
     
    transforms = [TRANSFORM_TYPE[i] for i in np.random.permutation(
        (np.linspace(start=0, stop=2, num=NUM_CFGS, dtype=float) >= 1.3).astype(int)
    )]
    
    inits = [INIT_TYPE[i] for i in np.random.permutation(
        np.linspace(start=0, stop=len(INIT_TYPE)-1, num=NUM_CFGS, dtype=int)
    )]
    
    lrs = np.random.permutation(
        np.linspace(start=LR[0], stop=LR[-1], num=NUM_CFGS, dtype=float)
    ).tolist()
    
    decays = np.random.permutation(
        np.linspace(start=W_DECAY[0], stop=W_DECAY[-1], num=NUM_CFGS, dtype=float)
    ).tolist()
    
    # Создание и сохранение конфигураций
    for i in range(NUM_CFGS):
        cfg = copy.deepcopy(exp_cfg)
        
        try:
            in_features = resolutions[i]['params']['size'][0]**2
            cfg.model_cfg.layers = generate_mlp_architecture(
                layers[i], in_features, 2
            )
            
            cfg.model_cfg.params.init_type = inits[i]
            
            cfg.data_cfg.train_transforms[0] = resolutions[i]
            cfg.data_cfg.train_transforms[-1] = transforms[i]
            
            cfg.data_cfg.eval_transforms[0] = resolutions[i]
            cfg.data_cfg.eval_transforms[-1] = transforms[i]
            
            cfg.train.learning_rate = lrs[i]
            cfg.train.weight_decay = decays[i]
            
            ckp_dir = os.path.join(cfg.checkpoints_dir, cls)
            os.makedirs(ckp_dir, exist_ok=True)

            cfg.checkpoints_dir = os.path.join(ckp_dir, f"ckp_{i}")

            # Data changes block
            cfg.expert_class = cls
            cfg.data_cfg.classes_num = 2

            cfg.data_cfg.label_mapping = {
                cls: 0,
                'unknown': 1
            }

            # Neptune
            cfg.neptune.tags = cls + '-' + str(i)
            
            write_file(cfg, os.path.join(save_path, f'cfg_{i}.pickle'))
            
        except Exception as e:
            print(f"Ошибка при генерации конфигурации {i}: {str(e)}")
            continue


if __name__ == '__main__':
    for cls in CLASSES:
        class_path = os.path.join(exp_cfg.configs_path, cls)
        os.makedirs(class_path, exist_ok=True)

        generate_config(class_path, cls)
