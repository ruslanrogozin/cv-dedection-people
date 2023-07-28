class Configs:
    path_data = 'data'
    path_new_data = 'new_data'
    device = 'cpu'
    image_loader_params = {'resize_size': 300,
                           'center_crop_cize': 300,
                           'normalize_mean': (0.5, 0.5, 0.5),
                           'normalize_std': (0.5, 0.5, 0.5)}
    model = {'decode_result': {'criteria': 0.8, 'max_output': 20},
             'normalize': {'mean': 128, 'std': 128}}
