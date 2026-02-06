import inspect
import timm.layers
from timm.models.vision_transformer import Attention

print("Attention init:", inspect.signature(Attention.__init__))
print("Attention forward:", inspect.signature(Attention.forward))

print("timm.layers members:", dir(timm.layers))
