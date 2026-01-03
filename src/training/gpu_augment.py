"""
GPU-side data augmentation for H100 optimization.

Moves MixUp and CutMix operations to GPU to avoid CPU bottlenecks.
Standard CPU-based augmentation causes data starvation on H100s.
"""

import torch
import torch.nn.functional as F


class GPUMixUp:
    """
    MixUp augmentation applied on GPU tensors.

    MixUp: x = λ * x_i + (1-λ) * x_j
           y = λ * y_i + (1-λ) * y_j

    Args:
        alpha: Beta distribution parameter (default 0.8)
        num_classes: Number of classes for one-hot encoding
    """

    def __init__(self, alpha=0.8, num_classes=10):
        self.alpha = alpha
        self.num_classes = num_classes

    def __call__(self, x, y):
        """
        Apply MixUp to batch.

        Args:
            x: Input tensor (B, C, H, W) on GPU
            y: Labels (B,) integers or (B, num_classes) one-hot on GPU

        Returns:
            mixed_x: Mixed inputs
            mixed_y: Mixed labels (soft targets)
        """
        if self.alpha <= 0:
            return x, y

        batch_size = x.size(0)
        device = x.device

        # Sample lambda from Beta distribution
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().to(device)

        # Random permutation for mixing
        index = torch.randperm(batch_size, device=device)

        # Mix inputs
        mixed_x = lam * x + (1 - lam) * x[index]

        # Convert labels to one-hot if needed
        if y.dim() == 1:
            y_onehot = F.one_hot(y, self.num_classes).float()
        else:
            y_onehot = y.float()

        # Mix labels
        mixed_y = lam * y_onehot + (1 - lam) * y_onehot[index]

        return mixed_x, mixed_y


class GPUCutMix:
    """
    CutMix augmentation applied on GPU tensors.

    CutMix: Cuts a patch from one image and pastes onto another.
    Label mixing is proportional to patch area.

    Args:
        alpha: Beta distribution parameter (default 1.0)
        num_classes: Number of classes for one-hot encoding
    """

    def __init__(self, alpha=1.0, num_classes=10):
        self.alpha = alpha
        self.num_classes = num_classes

    def _rand_bbox(self, size, lam, device):
        """Generate random bounding box."""
        W = size[2]
        H = size[3]

        # Ratio of cut area
        cut_rat = torch.sqrt(1.0 - lam)
        cut_w = (W * cut_rat).int()
        cut_h = (H * cut_rat).int()

        # Random center
        cx = torch.randint(0, W, (1,), device=device)
        cy = torch.randint(0, H, (1,), device=device)

        # Bounding box
        bbx1 = torch.clamp(cx - cut_w // 2, 0, W)
        bby1 = torch.clamp(cy - cut_h // 2, 0, H)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, W)
        bby2 = torch.clamp(cy + cut_h // 2, 0, H)

        return bbx1.item(), bby1.item(), bbx2.item(), bby2.item()

    def __call__(self, x, y):
        """
        Apply CutMix to batch.

        Args:
            x: Input tensor (B, C, H, W) on GPU
            y: Labels (B,) integers or (B, num_classes) one-hot on GPU

        Returns:
            mixed_x: CutMix inputs
            mixed_y: Mixed labels (soft targets)
        """
        if self.alpha <= 0:
            return x, y

        batch_size = x.size(0)
        device = x.device

        # Sample lambda from Beta distribution
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().to(device)

        # Random permutation for mixing
        index = torch.randperm(batch_size, device=device)

        # Get random bbox
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(x.size(), lam, device)

        # Apply CutMix
        mixed_x = x.clone()
        mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

        # Adjust lambda based on actual box area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))

        # Convert labels to one-hot if needed
        if y.dim() == 1:
            y_onehot = F.one_hot(y, self.num_classes).float()
        else:
            y_onehot = y.float()

        # Mix labels
        mixed_y = lam * y_onehot + (1 - lam) * y_onehot[index]

        return mixed_x, mixed_y


class GPUMixUpCutMix:
    """
    Combined MixUp and CutMix with random selection.

    Randomly applies either MixUp or CutMix (or neither) to each batch.

    Args:
        mixup_alpha: MixUp alpha parameter
        cutmix_alpha: CutMix alpha parameter
        mixup_prob: Probability of applying MixUp
        cutmix_prob: Probability of applying CutMix
        num_classes: Number of classes
    """

    def __init__(self, mixup_alpha=0.8, cutmix_alpha=1.0,
                 mixup_prob=0.5, cutmix_prob=0.5, num_classes=10):
        self.mixup = GPUMixUp(mixup_alpha, num_classes) if mixup_alpha > 0 else None
        self.cutmix = GPUCutMix(cutmix_alpha, num_classes) if cutmix_alpha > 0 else None
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob
        self.num_classes = num_classes

    def __call__(self, x, y):
        """
        Apply random augmentation to batch.

        Args:
            x: Input tensor (B, C, H, W) on GPU
            y: Labels on GPU

        Returns:
            augmented_x, augmented_y
        """
        # Randomly select augmentation
        r = torch.rand(1).item()

        if r < self.mixup_prob and self.mixup is not None:
            return self.mixup(x, y)
        elif r < self.mixup_prob + self.cutmix_prob and self.cutmix is not None:
            return self.cutmix(x, y)
        else:
            # No augmentation - convert labels to one-hot for consistency
            if y.dim() == 1:
                y = F.one_hot(y, self.num_classes).float()
            return x, y


class CUDAGraphWrapper:
    """
    CUDA Graph wrapper for training step optimization.

    CUDA Graphs capture and replay entire computation graphs,
    eliminating kernel launch overhead. Critical for small batches
    on H100 where GPU is faster than CPU scheduling.

    Requirements:
    - Fixed input shapes (use drop_last=True in DataLoader)
    - No dynamic control flow in forward pass
    - PyTorch 2.0+

    Usage:
        wrapper = CUDAGraphWrapper(model, optimizer, criterion, scaler, device)
        wrapper.warmup(sample_input, sample_target)
        wrapper.capture(sample_input, sample_target)

        for inputs, targets in dataloader:
            loss = wrapper.replay(inputs, targets)
    """

    def __init__(self, model, optimizer, criterion, scaler=None,
                 device='cuda', autocast_dtype=torch.bfloat16):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = scaler
        self.device = device
        self.autocast_dtype = autocast_dtype

        self.graph = None
        self.static_input = None
        self.static_target = None
        self.static_loss = None

    def warmup(self, sample_input, sample_target, num_warmup=3):
        """
        Warmup the CUDA stream before graph capture.

        Args:
            sample_input: Sample input tensor with correct shape
            sample_target: Sample target tensor with correct shape
            num_warmup: Number of warmup iterations
        """
        # Create static tensors
        self.static_input = sample_input.clone().to(self.device)
        self.static_target = sample_target.clone().to(self.device)

        # Warmup stream
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(s):
            for _ in range(num_warmup):
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', dtype=self.autocast_dtype):
                    output = self.model(self.static_input)
                    loss = self.criterion(output, self.static_target)
                loss.backward()
                self.optimizer.step()

        torch.cuda.current_stream().wait_stream(s)

    def capture(self, sample_input, sample_target):
        """
        Capture the training step as a CUDA Graph.

        Args:
            sample_input: Sample input tensor with correct shape
            sample_target: Sample target tensor with correct shape
        """
        if self.static_input is None:
            self.warmup(sample_input, sample_target)

        # Prepare for capture
        self.optimizer.zero_grad(set_to_none=True)

        # Capture graph
        self.graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self.graph):
            with torch.autocast(device_type='cuda', dtype=self.autocast_dtype):
                output = self.model(self.static_input)
                self.static_loss = self.criterion(output, self.static_target)
            self.static_loss.backward()
            self.optimizer.step()

    def replay(self, inputs, targets):
        """
        Replay the captured graph with new inputs.

        Args:
            inputs: New input tensor (must match captured shape)
            targets: New target tensor (must match captured shape)

        Returns:
            Loss value (from static tensor, updated after replay)
        """
        if self.graph is None:
            raise RuntimeError("Must call capture() before replay()")

        # Copy new data to static tensors
        self.static_input.copy_(inputs)
        self.static_target.copy_(targets)

        # Replay captured graph
        self.graph.replay()

        return self.static_loss.detach()

    def is_captured(self):
        """Check if graph has been captured."""
        return self.graph is not None


__all__ = ['GPUMixUp', 'GPUCutMix', 'GPUMixUpCutMix', 'CUDAGraphWrapper']
