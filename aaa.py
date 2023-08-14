from pathlib import Path

from config.config import Configs


def main():
    configs = Configs()
    work_directory = Path().cwd()
    device = configs.device

    #model = nvidia_ssd(
        #pretrained_default=True,
        #pretrainded_custom=False,
        #path=work_directory / configs.path_weight_model,
        #device=device
    #)
    from ssd.model import SSD300
    model = SSD300()
    model.eval()
    import torch
    inp = torch.rand((1,3,300,300))
    d1 = model(inp)
    print(d1[0].shape)


if __name__ == "__main__":
    main()
