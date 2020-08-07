script_list = {
    'ic17': ["Arabic", "Latin", "Chinese", "Japanese",
             "Korean", "Bangla", "Symbols"],
    'ic19': ["Arabic", "Latin", "Chinese", "Japanese",
             "Korean", "Bangla", "Hindi", "Symbols"]
}

script_weights = {
    'ic17rough': [1, 0.2, 1, 1, 1, 1, 1],
    'ic19rough': [1, 0.2, 1, 1, 1, 1, 1, 1]
}

norms = {
    'common': {
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5)
    },
    'icdar': {
        'mean': (.516, .497, .485),
        'std': (.171, .173, .157)
    },
    'imagenet': {
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225)
    }
}

config = {
    'num_workers': 16,
    'frequency': {
        'averagemeter': 100,  # the width of the AverageMeter
        'printloss': 100,
        'validate': 400
    },
    'data_dir': 'data/',
    'ckpt_dir': 'ckpt/',

    'input_size': 224,
    'transform': {
        'train': 'plain+',
        'val': 'plain+'
    },
    'scripts': 'ic17'
}
