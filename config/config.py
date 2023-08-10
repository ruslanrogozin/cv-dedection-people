class Configs:
    path_data = 'data'
    path_new_data = 'new_data'
    path_weight_model = 'weight'
    path_new_data = 'new_data'
    device = 'cuda'
    batch_size = 64
    use_padding_in_image_transform = False#True
    decode_result = {"criteria": 0.5, "max_output": 200, "pic_threshold": 0.3}
