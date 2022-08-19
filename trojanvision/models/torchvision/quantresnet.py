import torch.quantization

from resnet import _ResNet


class QuantResnet(_ResNet):

    def __init__(self):
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        super().__init__()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.quant(x)
        super(QuantResnet, self).forward()
        x = self.dequant(x)
