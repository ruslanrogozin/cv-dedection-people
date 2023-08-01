class Configs:
    path_data = 'data'
    path_new_data = 'new_data'
    path_weight_model = 'ssd'
    path_new_data = 'new_data'
    device = 'cpu'

    use_padding_in_image_transform = True

    image_loader_params = {'resize_size': 300,
                           # 'center_crop_cize': 300,
                           'normalize_mean': (0.5, 0.5, 0.5),
                           'normalize_std': (0.5, 0.5, 0.5)}
    decode_result = {'criteria': 0.5,
                     'max_output': 20,
                     'pic_threshold': 0.3
                     }
