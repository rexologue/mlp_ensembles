from enum import IntEnum

SetType = IntEnum('SetType', ('train', 'validation', 'test'))
SamplerType = IntEnum('SamplerType', ('Default', 'Upsampling'))
TransformType = IntEnum('TransformType', ('Resize', 'Normalize', 'Standardize', 'ToFloat'))
LayerType = IntEnum('LayerType', ('Linear', 'ReLU', 'LeakyReLU', 'Dropout'))
WeightsInitType = IntEnum('WeightsInitType', ('normal', 'uniform', 'he', 'xavier', 'xavier_normalized'))
