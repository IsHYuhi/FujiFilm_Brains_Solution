import math
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple, cast

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from .cam import GradCAM, GradCAMpp, SmoothGradCAMpp
from .HRNet import get_hrnet_config, get_seg_model


def fix_model_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    remove 'module.' of dataparallel
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith("module."):
            name = name[7:]
        new_state_dict[name] = v
    return new_state_dict


def weights_init(init_type: str = "gaussian") -> Callable[[nn.Module], None]:
    def init_fun(m: nn.Module) -> None:
        classname = m.__class__.__name__
        if (classname.find("Conv") == 0 or classname.find("Linear") == 0) and hasattr(
            m, "weight"
        ):
            if init_type == "gaussian":
                nn.init.normal_(cast(torch.Tensor, m.weight), 0.0, 0.02)
            elif init_type == "xavier":
                nn.init.xavier_normal_(cast(torch.Tensor, m.weight), gain=math.sqrt(2))
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(cast(torch.Tensor, m.weight), a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(cast(torch.Tensor, m.weight), gain=math.sqrt(2))
            elif init_type == "default":
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(cast(torch.Tensor, m.bias), 0.0)

    return init_fun


class FCNet(nn.Module):
    def __init__(
        self, pretrained: bool = False, in_channels: int = 3, out_channels: int = 2
    ) -> None:
        super(FCNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        for conv in self.features:
            conv.apply(weights_init("gaussian"))

        self.global_maxpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_maxpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_fcn(**kwargs: Any) -> FCNet:
    model = FCNet(**kwargs)
    return model


def grad_cam_fcn(
    pretrained: bool = False, path: str = "default", **kwargs: Any
) -> FCNet:
    model = FCNet(**kwargs)
    if pretrained:
        state_dict = torch.load(path)  # map_location
        model.load_state_dict(fix_model_state_dict(state_dict))
        print("loaded")
    model.eval()
    target_layer = model.features[3]
    wrapped_model = GradCAM(model, target_layer)
    return wrapped_model


def grad_campp_fcn(
    pretrained: bool = False, path: str = "default", **kwargs: Any
) -> FCNet:
    model = FCNet(**kwargs)
    if pretrained:
        state_dict = torch.load(path)  # map_location
        model.load_state_dict(fix_model_state_dict(state_dict))
        print("loaded")
    model.eval()
    target_layer = model.features[3]
    wrapped_model = GradCAMpp(model, target_layer)
    return wrapped_model


def smooth_grad_cam_fcn(
    pretrained: bool = False, path: str = "default", **kwargs: Any
) -> FCNet:
    model = FCNet(**kwargs)
    if pretrained:
        state_dict = torch.load(path)  # map_location
        model.load_state_dict(fix_model_state_dict(state_dict))
        print("loaded")
    model.eval()
    target_layer = model.features[3]
    wrapped_model = SmoothGradCAMpp(
        model, target_layer, n_samples=25, stdev_spread=0.15
    )
    return wrapped_model


def get_model(
    model_name: str,
    encoder_name: str,
    weight_name: Optional[str] = None,
    in_channels: int = 3,
    classes: int = 1,
    activation: str = "sigmoid",
) -> Tuple[nn.Module, Optional[str]]:
    if model_name is None:
        assert "please input model name"
        exit()

    if model_name == "Unet":
        net = smp.Unet(
            in_channels=in_channels,
            encoder_name=encoder_name,
            classes=classes,
            activation=activation,
            encoder_weights=None,
        )
    elif model_name == "UnetPlusPlus":
        net = smp.UnetPlusPlus(
            in_channels=in_channels,
            encoder_name=encoder_name,
            classes=classes,
            activation=activation,
            encoder_weights=None,
        )
    elif model_name == "MAnet":
        net = smp.MAnet(
            in_channels=in_channels,
            encoder_name=encoder_name,
            classes=classes,
            activation=activation,
            encoder_weights=None,
        )
    elif model_name == "Linknet":
        net = smp.Linknet(
            in_channels=in_channels,
            encoder_name=encoder_name,
            classes=classes,
            activation=activation,
            encoder_weights=None,
        )
    elif model_name == "FPN":
        net = smp.FPN(
            in_channels=in_channels,
            encoder_name=encoder_name,
            classes=classes,
            activation=activation,
            encoder_weights=None,
        )
    elif model_name == "PSPNet":
        net = smp.PSPNet(
            in_channels=in_channels,
            encoder_name=encoder_name,
            classes=classes,
            activation=activation,
            encoder_weights=None,
        )
    elif model_name == "PAN":
        net = smp.PAN(
            in_channels=in_channels,
            encoder_name=encoder_name,
            classes=classes,
            activation=activation,
            encoder_weights=None,
        )
    elif model_name == "DeepLabV3":
        net = smp.DeepLabV3(
            in_channels=in_channels,
            encoder_name=encoder_name,
            classes=classes,
            activation=activation,
            encoder_weights=None,
        )
    elif model_name == "DeepLabV3Plus":
        net = smp.DeepLabV3Plus(
            in_channels=in_channels,
            encoder_name=encoder_name,
            classes=classes,
            activation=activation,
            encoder_weights=None,
        )
    elif model_name == "HighResolutionNet":
        net = get_seg_model(get_hrnet_config())
        weight_name = None
    else:
        assert "please input correct model name"
        exit()

    return net, weight_name
