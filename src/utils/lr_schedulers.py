import math

class LinearWarmupCosineDecay:
    def __init__(
        self,
        warmup_steps: int,
        total_steps: int,
        final_lr_ratio: float,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.final_lr_ratio = final_lr_ratio

    def __call__(self, current_step: int) -> float:
        if current_step < self.warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, self.warmup_steps))
        
        # Cosine decay
        progress = float(current_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        progress = min(1.0, max(0.0, progress)) # Clip to [0, 1]
        
        # Cosine decay from 1.0 to final_lr_ratio
        # formula: final + 0.5 * (initial - final) * (1 + cos(pi * progress))
        # scaled relative to initial lr (which is 1.0 in lambda)
        
        cosine_part = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.final_lr_ratio + (1.0 - self.final_lr_ratio) * cosine_part
