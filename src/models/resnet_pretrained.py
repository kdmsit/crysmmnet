import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


# class ResNet(nn.Module):
#
#     def __init__(self, base_model,out_dim):
#         super(ResNet, self).__init__()
#         self.resnet_dict = {"resnet18": models.resnet18(pretrained=False,num_classes=out_dim),
#                             "resnet50": models.resnet50(pretrained=False,num_classes=out_dim)}
#
#         self.backbone = self._get_basemodel(base_model)
#         dim_mlp = self.backbone.fc.in_features
#
#         # add mlp projection head
#         self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)
#
#     def _get_basemodel(self, model_name):
#             pretrained_resnet=models.resnet18(pretrained=True)
#             model = self.resnet_dict[model_name]
#             return model
#
#     def forward(self, x):
#         return self.backbone(x)


class ResNet_Pretrained(nn.Module):
    def __init__(self, output_layer=None):
        super().__init__()
        self.pretrained = models.resnet18(pretrained=True)
        self.output_layer = output_layer
        self.layers = list(self.pretrained._modules.keys())
        self.layer_count = 0
        for l in self.layers:
            if l != self.output_layer:
                self.layer_count += 1
            else:
                break
        for i in range(1, len(self.layers) - self.layer_count):
            self.dummy_var = self.pretrained._modules.pop(self.layers[-i])

        self.net = nn.Sequential(self.pretrained._modules)
        self.pretrained = None
        self.linear = nn.Linear(512 , 32)

    def forward(self, x):
        x = self.net(x)
        out = F.avg_pool2d(x, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out