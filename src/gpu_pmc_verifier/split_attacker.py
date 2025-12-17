import torch
import torch.nn as nn
import random
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import math


@dataclass
class KernelShape:
    """Shape of a kernel in the format (B, M, K, N) for matrix multiplication."""

    b: int  # batch size
    m: int  # rows of first matrix
    k: int  # columns of first matrix / rows of second matrix
    n: int  # columns of second matrix


class SplitLinearLayer(nn.Module):
    """A linear layer that can be split into multiple sub-layers."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        device: torch.device,
        granularity: int = 256,
        split: int = 1,
        split_type: Optional[str] = None,
        weight: Optional[torch.Tensor] = None,
    ):
        """Initialize the split linear layer.

        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            device: Device to run operations on
            granularity: Minimum block size for matrix operations
            split: Number of splits to apply
            split_type: Type of split to apply (vstack, hstack, k-split, batch-split)
            weight: Optional weight tensor to initialize the layer
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.granularity = granularity
        self.split = split
        self.split_type = split_type
        self.kernels: List[KernelShape] = []

        # Validate dimensions
        if not (in_dim < granularity or in_dim % granularity == 0):
            raise ValueError(
                f"Dimension {in_dim} must be either < {granularity} or divisible by it"
            )
        if not (out_dim < granularity or out_dim % granularity == 0):
            raise ValueError(
                f"Dimension {out_dim} must be either < {granularity} or divisible by it"
            )

        # Initialize base layer
        self.base_layer = nn.Linear(in_dim, out_dim, bias=False)
        if weight is not None:
            self.base_layer.weight.data = weight.clone()
        self.base_layer = self.base_layer.to(device)

        # Initialize split layers if needed
        self.split_layers: List[nn.Linear] = []
        self.split_sizes: List[List[int]] = []
        if split > 1:
            self._initialize_split_layers()

    def _initialize_split_layers(self):
        """Initialize the split layers based on the split type."""
        if (self.split == 2 and self.split_type == "hstack") or (
            self.split == 4 and self.split_type == "b-hstack"
        ):
            # Split along output dimension
            split_sizes = self._get_random_split_sizes(self.out_dim, 2)
            self.split_sizes = [split_sizes]
            weight_blocks = torch.split(self.base_layer.weight.data, split_sizes, dim=0)
            for size, block in zip(split_sizes, weight_blocks):
                layer = nn.Linear(self.in_dim, size, bias=False)
                layer.weight.data = block
                self.split_layers.append(layer.to(self.device))

        elif (
            (self.split == 2 and self.split_type == "k-split")
            or (self.split == 4 and self.split_type == "A-block")
            or (self.split == 4 and self.split_type == "b-k-split")
            or (self.split == 8 and self.split_type == "b-A-block")
        ):
            # Split along input dimension
            split_sizes = self._get_random_split_sizes(self.in_dim, 2)
            self.split_sizes = [split_sizes]
            weight_blocks = torch.split(self.base_layer.weight.data, split_sizes, dim=1)
            for size, block in zip(split_sizes, weight_blocks):
                layer = nn.Linear(size, self.out_dim, bias=False)
                layer.weight.data = block
                self.split_layers.append(layer.to(self.device))

        elif (
            (self.split == 4 and self.split_type == "B-block")
            or (self.split == 8 and self.split_type == "full-block")
            or (self.split == 8 and self.split_type == "b-B-block")
            or (self.split == 16 and self.split_type == "b-full-block")
        ):
            in_dim_splits = self._get_random_split_sizes(self.in_dim, 2)
            out_dim_splits = self._get_random_split_sizes(self.out_dim, 2)
            self.split_sizes = [in_dim_splits, out_dim_splits]
            weight_blocks = []
            for weight_row in torch.split(
                self.base_layer.weight.data, in_dim_splits, dim=1
            ):
                weight_blocks.extend(torch.split(weight_row, out_dim_splits, dim=0))

            for block in weight_blocks:
                layer = nn.Linear(block.shape[1], block.shape[0], bias=False)
                layer.weight.data = block
                self.split_layers.append(layer.to(self.device))

    def _get_random_split_sizes(self, total_size: int, split_count: int) -> List[int]:
        """Generate random splits that respect granularity rule."""
        if not (total_size < self.granularity or total_size % self.granularity == 0):
            raise ValueError(
                f"Size {total_size} must be either < {self.granularity} or divisible by it"
            )

        if total_size < self.granularity:
            # For small sizes, just split evenly
            base_size = total_size // split_count
            remainder = total_size % split_count
            return [base_size + (1 if i < remainder else 0) for i in range(split_count)]

        # For large sizes, split in granularity units
        base_units = total_size // self.granularity
        dividers = sorted(random.sample(range(1, base_units), k=split_count - 1))
        dividers = [0] + dividers + [base_units]
        split_units = [dividers[i + 1] - dividers[i] for i in range(split_count)]
        return [u * self.granularity for u in split_units]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the split linear layer.

        Args:
            x: Input tensor of shape (batch, seq_len, in_dim)

        Returns:
            Output tensor after applying the split strategy
        """
        batch, seq_len, in_dim = x.shape

        if self.split == 1:
            # No split
            result = self.base_layer(x)
            self.kernels.append(KernelShape(batch, seq_len, in_dim, self.out_dim))
            return result

        if self.split == 2 and self.split_type == "vstack":
            # Split along sequence length
            split_sizes = self._get_random_split_sizes(seq_len, self.split)
            x_blocks = torch.split(x, split_sizes, dim=1)
            layer = self.base_layer
            results = []
            for xi in x_blocks:
                y = layer(xi)
                self.kernels.append(
                    KernelShape(batch, xi.shape[1], in_dim, layer.out_features)
                )
                results.append(y)
            return torch.cat(results, dim=1)

        elif self.split == 2 and self.split_type == "hstack":
            # Split along output dimension
            results = []
            for layer in self.split_layers:
                y = layer(x)
                self.kernels.append(
                    KernelShape(batch, seq_len, in_dim, layer.out_features)
                )
                results.append(y)
            return torch.cat(results, dim=2)

        elif self.split == 2 and self.split_type == "k-split":
            # Split along input dimension
            split_sizes = self.split_sizes[0]
            x_blocks = torch.split(x, split_sizes, dim=2)
            result = torch.zeros(batch, seq_len, self.out_dim, device=self.device)
            for xi, layer in zip(x_blocks, self.split_layers):
                y = layer(xi)
                self.kernels.append(
                    KernelShape(batch, seq_len, xi.shape[2], self.out_dim)
                )
                result += y
            return result

        elif self.split == 2 and self.split_type == "batch-split":
            # Split along batch dimension
            split_sizes = self._get_random_split_sizes(batch, self.split)
            x_blocks = torch.split(x, split_sizes, dim=0)
            layer = self.base_layer
            results = []
            for xi in x_blocks:
                y = layer(xi)
                self.kernels.append(
                    KernelShape(xi.shape[0], seq_len, in_dim, self.out_dim)
                )
                results.append(y)
            return torch.cat(results, dim=0)

        elif self.split == 4 and self.split_type == "A-block":
            # Split A into 2x2 blocks and B horizontally
            seq_splits = self._get_random_split_sizes(seq_len, 2)
            in_dim_splits = self.split_sizes[0]
            x_blocks = []
            for x_row in torch.split(x, seq_splits, dim=1):
                x_blocks.extend(torch.split(x_row, in_dim_splits, dim=2))
            result = torch.zeros(batch, seq_len, self.out_dim, device=self.device)
            for i, xi in enumerate(x_blocks):
                y = self.split_layers[i % 2](xi)
                m_start = sum(seq_splits[: i // 2])
                m_end = m_start + xi.shape[1]
                result[:, m_start:m_end, :] += y
                self.kernels.append(
                    KernelShape(batch, xi.shape[1], xi.shape[2], self.out_dim)
                )
            return result

        elif self.split == 4 and self.split_type == "B-block":
            # Split B into 2x2 blocks and A vertically
            in_dim_splits = self.split_sizes[0]
            out_dim_splits = self.split_sizes[1]
            x_blocks = torch.split(x, in_dim_splits, dim=2)
            results = []
            for i in range(2):
                subres = torch.zeros(
                    batch, seq_len, out_dim_splits[i], device=self.device
                )
                for j in range(2):
                    subres += self.split_layers[i + 2 * j](x_blocks[j])
                    self.kernels.append(
                        KernelShape(batch, seq_len, in_dim_splits[j], out_dim_splits[i])
                    )
                results.append(subres)
            return torch.cat(results, dim=2)

        elif self.split == 4 and self.split_type == "b-vstack":
            # Split along batch dimension and input dimension
            batch_splits = self._get_random_split_sizes(batch, 2)
            seq_splits = self._get_random_split_sizes(seq_len, 2)
            x_batch_blocks = torch.split(x, batch_splits, dim=0)
            layer = self.base_layer
            results = []
            for x_batch_block in x_batch_blocks:
                x_blocks = torch.split(x_batch_block, seq_splits, dim=1)
                subres = []
                for x_block in x_blocks:
                    y = layer(x_block)
                    self.kernels.append(
                        KernelShape(
                            x_block.shape[0],
                            x_block.shape[1],
                            in_dim,
                            layer.out_features,
                        )
                    )
                    subres.append(y)
                results.append(torch.cat(subres, dim=1))
            return torch.cat(results, dim=0)

        elif self.split == 4 and self.split_type == "b-hstack":
            # Split along batch dimension and output dimension
            batch_splits = self._get_random_split_sizes(batch, 2)
            out_dim_splits = self.split_sizes[0]
            x_blocks = torch.split(x, batch_splits, dim=0)
            results = []
            for x_block in x_blocks:
                subres = []
                for layer in self.split_layers:
                    y = layer(x_block)
                    self.kernels.append(
                        KernelShape(
                            x_block.shape[0], seq_len, in_dim, layer.out_features
                        )
                    )
                    subres.append(y)
                results.append(torch.cat(subres, dim=2))
            return torch.cat(results, dim=0)

        elif self.split == 4 and self.split_type == "b-k-split":
            # Split along batch dimension and input dimension
            batch_splits = self._get_random_split_sizes(batch, 2)
            in_dim_splits = self.split_sizes[0]
            x_batch_blocks = torch.split(x, batch_splits, dim=0)
            results = []
            for x_batch_block in x_batch_blocks:
                x_blocks = torch.split(x_batch_block, in_dim_splits, dim=2)
                subres = torch.zeros(
                    batch_splits[0], seq_len, self.out_dim, device=self.device
                )
                for x_block, layer in zip(x_blocks, self.split_layers):
                    y = layer(x_block)
                    self.kernels.append(
                        KernelShape(
                            x_block.shape[0], seq_len, x_block.shape[2], self.out_dim
                        )
                    )
                    subres += y
                results.append(subres)
            return torch.cat(results, dim=0)

        elif self.split == 8 and self.split_type == "full-block":
            # Split along batch dimension and input dimension
            seq_splits = self._get_random_split_sizes(seq_len, 2)
            in_dim_splits = self.split_sizes[0]
            out_dim_splits = self.split_sizes[1]
            x_blocks = []
            for x_row in torch.split(x, seq_splits, dim=1):
                x_blocks.extend(torch.split(x_row, in_dim_splits, dim=2))
            results = torch.zeros(batch, seq_len, self.out_dim, device=self.device)
            for i in range(4):  # 4 blocks in A
                x_block = x_blocks[i]
                for j in range(4):  # 4 blocks in B
                    layer = self.split_layers[j]
                    if i % 2 == j // 2:
                        y = layer(x_block)
                        m_start = sum(seq_splits[: i // 2])
                        m_end = m_start + x_block.shape[1]
                        n_start = sum(out_dim_splits[: j % 2])
                        n_end = n_start + y.shape[2]
                        results[:, m_start:m_end, n_start:n_end] += y
                        self.kernels.append(
                            KernelShape(
                                batch, x_block.shape[1], x_block.shape[2], y.shape[2]
                            )
                        )
            return results

        elif self.split == 8 and self.split_type == "b-A-block":
            # Split along batch dimension and A into 2x2 blocks
            batch_splits = self._get_random_split_sizes(batch, 2)
            seq_splits = self._get_random_split_sizes(seq_len, 2)
            in_dim_splits = self.split_sizes[0]
            x_batch_blocks = torch.split(x, batch_splits, dim=0)
            results = []
            for x_batch_block in x_batch_blocks:
                x_blocks = []
                for x_row in torch.split(x_batch_block, seq_splits, dim=1):
                    x_blocks.extend(torch.split(x_row, in_dim_splits, dim=2))
                subres = torch.zeros(
                    batch_splits[0], seq_len, self.out_dim, device=self.device
                )
                for i, xi in enumerate(x_blocks):
                    y = self.split_layers[i % 2](xi)
                    m_start = sum(seq_splits[: i // 2])
                    m_end = m_start + xi.shape[1]
                    subres[:, m_start:m_end, :] += y
                    self.kernels.append(
                        KernelShape(xi.shape[0], xi.shape[1], xi.shape[2], self.out_dim)
                    )
                results.append(subres)
            return torch.cat(results, dim=0)

        elif self.split == 8 and self.split_type == "b-B-block":
            # Split along batch dimension and B into 2x2 blocks
            batch_splits = self._get_random_split_sizes(batch, 2)
            in_dim_splits = self.split_sizes[0]
            out_dim_splits = self.split_sizes[1]
            x_batch_blocks = torch.split(x, batch_splits, dim=0)
            results = []
            for x_batch_block in x_batch_blocks:
                x_blocks = torch.split(x_batch_block, in_dim_splits, dim=2)
                subres = []
                for i in range(2):
                    subsubres = torch.zeros(
                        batch_splits[0], seq_len, out_dim_splits[i], device=self.device
                    )
                    for j in range(2):
                        subsubres += self.split_layers[i + 2 * j](x_blocks[j])
                        self.kernels.append(
                            KernelShape(
                                batch_splits[0],
                                seq_len,
                                in_dim_splits[j],
                                out_dim_splits[i],
                            )
                        )
                    subres.append(subsubres)
                results.append(torch.cat(subres, dim=2))
            return torch.cat(results, dim=0)

        elif self.split == 16 and self.split_type == "b-full-block":
            # Split along batch dimension and input dimension
            batch_splits = self._get_random_split_sizes(batch, 2)
            seq_splits = self._get_random_split_sizes(seq_len, 2)
            in_dim_splits = self.split_sizes[0]
            out_dim_splits = self.split_sizes[1]
            x_batch_blocks = torch.split(x, batch_splits, dim=0)
            results = []
            for x_batch_block in x_batch_blocks:
                x_blocks = []
                for x_row in torch.split(x_batch_block, seq_splits, dim=1):
                    x_blocks.extend(torch.split(x_row, in_dim_splits, dim=2))
                subres = torch.zeros(
                    x_batch_block.shape[0], seq_len, self.out_dim, device=self.device
                )
                for i in range(4):
                    x_block = x_blocks[i]
                    for j in range(4):
                        layer = self.split_layers[j]
                        if i % 2 == j // 2:
                            y = layer(x_block)
                            m_start = sum(seq_splits[: i // 2])
                            m_end = m_start + x_block.shape[1]
                            n_start = sum(out_dim_splits[: j % 2])
                            n_end = n_start + y.shape[2]
                            subres[:, m_start:m_end, n_start:n_end] += y
                            self.kernels.append(
                                KernelShape(
                                    x_batch_block.shape[0],
                                    x_block.shape[1],
                                    x_block.shape[2],
                                    y.shape[2],
                                )
                            )
                results.append(subres)
            return torch.cat(results, dim=0)

        else:
            raise ValueError(f"Invalid split type: ({self.split}, {self.split_type})")

    def get_kernels(self) -> List[KernelShape]:
        """Get the list of kernel shapes used in the forward pass."""
        return self.kernels

    def clear_kernels(self) -> None:
        """Clear the list of kernel shapes."""
        self.kernels = []


class TransformerLayer:
    """Represents a transformer layer with multiple linear operations that can be split."""

    # Define standard transformer layer names in order
    LAYER_NAMES = [
        "q_proj",  # Query projection
        "k_proj",  # Key projection
        "v_proj",  # Value projection
        "o_proj",  # Output projection
        "up_proj",  # FFN up projection
        "down_proj",  # FFN down projection
    ]

    def __init__(self, device: torch.device, granularity: int = 256):
        """Initialize the transformer layer.

        Args:
            device: Device to run operations on
            granularity: Minimum block size for matrix operations
        """
        self.device = device
        self.granularity = granularity
        self.linear_layers: Dict[str, SplitLinearLayer] = {}
        self.kernels: Dict[str, List[KernelShape]] = {
            name: [] for name in self.LAYER_NAMES
        }

    def add_linear(
        self,
        name: str,
        in_dim: int,
        out_dim: int,
        split: int = 1,
        split_type: Optional[str] = None,
        weight: Optional[torch.Tensor] = None,
    ) -> None:
        """Add a linear layer to the transformer.

        Args:
            name: Name of the linear layer (must be one of LAYER_NAMES)
            in_dim: Input dimension
            out_dim: Output dimension
            split: Number of splits to apply
            split_type: Type of split to apply (vstack, hstack, k-split, batch-split)
            weight: Optional weight tensor to initialize the layer
        """
        if name not in self.LAYER_NAMES:
            raise ValueError(f"Layer name must be one of {self.LAYER_NAMES}")
        if name in self.linear_layers:
            raise ValueError(f"Layer {name} already exists")

        layer = SplitLinearLayer(
            in_dim=in_dim,
            out_dim=out_dim,
            device=self.device,
            granularity=self.granularity,
            split=split,
            split_type=split_type,
            weight=weight,
        )
        self.linear_layers[name] = layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer layer.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)

        Returns:
            Output tensor after all operations
        """
        residual = x
        # Self-attention projections
        q = self.linear_layers["q_proj"](x)  # (batch, seq_len, hidden_dim)
        k = self.linear_layers["k_proj"](x)  # (batch, seq_len, hidden_dim)
        v = self.linear_layers["v_proj"](x)  # (batch, seq_len, hidden_dim)

        # Collect kernels from attention projections
        for layer_name in ["q_proj", "k_proj", "v_proj"]:
            layer = self.linear_layers[layer_name]
            self.kernels[layer_name].extend(layer.get_kernels())
            layer.clear_kernels()

        # Attention computation
        # QK^T
        attn = torch.matmul(q, k.transpose(-2, -1))  # (batch, seq_len, seq_len)
        attn = attn / math.sqrt(q.size(-1))  # Scale by sqrt(d_k)
        attn = torch.softmax(attn, dim=-1)  # (batch, seq_len, seq_len)

        # Attention output
        attn_out = torch.matmul(attn, v)  # (batch, seq_len, hidden_dim)

        # Output projection
        x = self.linear_layers["o_proj"](attn_out)
        self.kernels["o_proj"].extend(self.linear_layers["o_proj"].get_kernels())
        self.linear_layers["o_proj"].clear_kernels()

        x = x + residual
        residual = x

        # FFN
        x = self.linear_layers["up_proj"](x)
        x = torch.nn.functional.gelu(x)
        x = self.linear_layers["down_proj"](x)

        # Collect kernels from FFN
        for layer_name in ["up_proj", "down_proj"]:
            layer = self.linear_layers[layer_name]
            self.kernels[layer_name].extend(layer.get_kernels())
            layer.clear_kernels()

        x = x + residual

        return x

    def get_kernels(self) -> Dict[str, List[KernelShape]]:
        """Get the dictionary of kernel shapes used in the forward pass."""
        return self.kernels

    def clear_kernels(self) -> None:
        """Clear all kernel shapes."""
        self.kernels = {name: [] for name in self.LAYER_NAMES}


class SplitAttacker:
    """Attempts matrix splitting attacks to evade upper bound check."""

    def __init__(self, seed: int = 0, granularity: int = 256):
        """Initialize the split attacker with a random seed and granularity.

        Args:
            seed: Random seed for reproducibility
            granularity: Minimum block size for matrix operations (default: 256)
        """
        torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.granularity = granularity
        print(f"Using device: {self.device}, granularity: {granularity}")

    def _init_transformer_with_splits(
        self,
        d: int,
        d_ffn: int,
        layer_splits: Dict[str, Tuple[int, str]],
    ) -> TransformerLayer:
        """
        Initialize a transformer layer with given splits for each layer.

        Args:
            d: Hidden dimension
            d_ffn: Feed-forward network dimension
            layer_splits: Dictionary mapping layer names to (split, split_type) tuples

        Returns:
            Initialized transformer layer
        """
        transformer = TransformerLayer(self.device, self.granularity)

        # Add attention layers with their splits
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            split, split_type = layer_splits[name]
            transformer.add_linear(
                name,
                d,
                d,
                split=split,
                split_type=split_type,
            )

        # Add FFN layers with their splits
        split, split_type = layer_splits["up_proj"]
        transformer.add_linear(
            "up_proj",
            d,
            d_ffn,
            split=split,
            split_type=split_type,
        )

        split, split_type = layer_splits["down_proj"]
        transformer.add_linear(
            "down_proj",
            d_ffn,
            d,
            split=split,
            split_type=split_type,
        )

        return transformer

    def _sample_layer_splits(
        self,
        b: int,
        seq_len: int,
        d: int,
    ) -> Dict[str, Tuple[int, str]]:
        """
        Sample splits and types for each layer based on dimensions.

        Args:
            b: Batch size
            seq_len: Sequence length
            d: Hidden dimension

        Returns:
            Dictionary mapping layer names to (split, split_type) tuples
        """
        # Define all possible split combinations
        split_combinations = [
            (1, "none"),  # No split
            (2, "vstack"),
            (2, "hstack"),
            (2, "k-split"),
            (2, "batch-split"),
            # (4, "A-block"),
            # (4, "B-block"),
            # (4, "b-vstack"),
            # (4, "b-hstack"),
            # (4, "b-k-split"),
            # (8, "full-block"),
            # (8, "b-A-block"),
            # (8, "b-B-block"),
            # (16, "b-full-block"),
        ]

        # Sample split and type for each layer
        layer_splits = {}
        for layer_name in TransformerLayer.LAYER_NAMES:
            # Determine valid split types based on dimensions
            valid_combinations = []
            for split, split_type in split_combinations:
                if split == 1:  # No split is always valid
                    valid_combinations.append((split, split_type))
                elif split_type == "vstack" and seq_len > 1:
                    valid_combinations.append((split, split_type))
                elif split_type == "hstack" and d > 1:
                    valid_combinations.append((split, split_type))
                elif split_type == "k-split" and d > 1:
                    valid_combinations.append((split, split_type))
                elif split_type == "batch-split" and b > 1:
                    valid_combinations.append((split, split_type))
                elif split_type == "A-block" and seq_len > 1 and d > 1:
                    valid_combinations.append((split, split_type))
                elif split_type == "B-block" and d > 1:
                    valid_combinations.append((split, split_type))
                elif split_type == "b-vstack" and b > 1 and seq_len > 1:
                    valid_combinations.append((split, split_type))
                elif split_type == "b-hstack" and b > 1 and d > 1:
                    valid_combinations.append((split, split_type))
                elif split_type == "b-k-split" and b > 1 and d > 1:
                    valid_combinations.append((split, split_type))
                elif split_type == "full-block" and seq_len > 1 and d > 1:
                    valid_combinations.append((split, split_type))
                elif split_type == "b-A-block" and b > 1 and seq_len > 1 and d > 1:
                    valid_combinations.append((split, split_type))
                elif split_type == "b-B-block" and b > 1 and d > 1:
                    valid_combinations.append((split, split_type))
                elif split_type == "b-full-block" and b > 1 and seq_len > 1 and d > 1:
                    valid_combinations.append((split, split_type))

            if not valid_combinations:
                raise ValueError(
                    f"No valid split combinations for dimensions: b={b}, seq_len={seq_len}, d={d}"
                )

            # Sample a valid combination
            layer_splits[layer_name] = random.choice(valid_combinations)

        return layer_splits

    def attempt_attack(
        self,
        b: int,
        seq_len: int,
        d: int,
        d_ffn: int,
        test_mode: bool = False,
    ) -> Tuple[Dict[str, List[KernelShape]], torch.Tensor]:
        """
        Attempt matrix splitting attacks to evade upper bound check.

        Args:
            b: Batch size
            seq_len: Sequence length
            d: Hidden dimension
            d_ffn: Feed-forward network dimension
            test_mode: If True, calculate reference results and verify correctness

        Returns:
            Tuple of (dictionary mapping layer names to their kernel shapes, output tensor)
        """
        # Sample splits for all layers
        layer_splits = self._sample_layer_splits(b, seq_len, d)

        # Create transformer layer with sampled splits
        transformer = self._init_transformer_with_splits(d, d_ffn, layer_splits)

        # Create input tensor
        x = torch.randn(b, seq_len, d, device=self.device)

        # Calculate reference result if in test mode
        reference_result = None
        if test_mode:
            # Create a new transformer without splits for reference
            ref_transformer = self._init_transformer_with_splits(
                d, d_ffn, {name: (1, "none") for name in TransformerLayer.LAYER_NAMES}
            )
            # Copy weights from test transformer to reference transformer
            for layer_name in TransformerLayer.LAYER_NAMES:
                ref_transformer.linear_layers[
                    layer_name
                ].base_layer.weight.data = transformer.linear_layers[
                    layer_name
                ].base_layer.weight.data.clone()
            reference_result = ref_transformer.forward(x)

        try:
            # Run forward pass
            result = transformer.forward(x)

            # Verify result if in test mode
            if test_mode and reference_result is not None:
                # Calculate relative tolerance based on magnitude of reference result
                max_abs_ref = torch.max(torch.abs(reference_result)).item()
                rtol = max(1e-5, 1e-5 * max_abs_ref)  # Scale tolerance with magnitude
                atol = max(1e-5, 1e-5 * max_abs_ref)  # Scale absolute tolerance too

                # Verify result matches reference
                if not torch.allclose(result, reference_result, rtol=rtol, atol=atol):
                    # Find max difference and its location
                    diff = torch.abs(result - reference_result)
                    max_diff, max_idx = (
                        torch.max(diff.view(-1)),
                        torch.argmax(diff.view(-1)),
                    )
                    # For 3D tensors, unravel the flat index
                    max_b, max_i, max_j = torch.unravel_index(max_idx, result.shape)

                    print("Result does not match reference")
                    print(
                        f"Max difference: {max_diff} at position (batch={max_b}, i={max_i}, j={max_j})"
                    )
                    print(f"Result value: {result[max_b, max_i, max_j]}")
                    print(f"Reference value: {reference_result[max_b, max_i, max_j]}")
                    print(
                        f"Relative difference: {max_diff / (torch.abs(reference_result[max_b, max_i, max_j]) + 1e-10)}"
                    )
                    print(f"Used tolerance - rtol: {rtol}, atol: {atol}")
                    print(f"Layer splits: {layer_splits}")
                    raise ValueError("Result does not match reference")

        except ValueError as e:
            print(f"Error during forward pass: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

        print(
            f"Total kernels tested: {sum(len(kernels) for kernels in transformer.get_kernels().values())}"
        )
        for layer_name, kernels in transformer.get_kernels().items():
            split, split_type = layer_splits[layer_name]
            print(
                f"  {layer_name}: split={split} type={split_type} with {len(kernels)} kernels:"
            )
            for kernel in kernels:
                print(
                    f"    - KernelShape(b={kernel.b}, m={kernel.m}, k={kernel.k}, n={kernel.n})"
                )

        return transformer.get_kernels(), result
