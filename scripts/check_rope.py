import inspect
from timm.models.vision_transformer import Attention, Block

print("Attention init:", inspect.signature(Attention.__init__))
print("Attention forward:", inspect.signature(Attention.forward))

# Check if there are any RoPE related args
import timm.layers
print("timm.layers members:", dir(timm.layers))
