import inspect
from timm.models.vision_transformer import Block

print(inspect.signature(Block.__init__))
