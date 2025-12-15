import inspect
from timm.layers import RotaryEmbedding

print(inspect.signature(RotaryEmbedding.__init__))
print(inspect.signature(RotaryEmbedding.forward))
