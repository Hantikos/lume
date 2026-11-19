"""Neural-network runtime for Lume: tensors, layers, optimizers, losses, metrics.

Everything here is implemented in plain Python so the language stays dependency
free.  ``numpy`` is used automatically when it happens to be installed, but it is
never required.
"""

from __future__ import annotations

import json
import math
import random

try:  # optional acceleration, never required
    import numpy as _np
except Exception:  # pragma: no cover - numpy is optional
    _np = None


# --------------------------------------------------------------------------
# Tensor
# --------------------------------------------------------------------------
class Tensor:
    """A small dense matrix (rows x cols) of floats."""

    __slots__ = ("rows", "cols", "data")

    def __init__(self, rows: int, cols: int, data=None):
        self.rows = rows
        self.cols = cols
        if data is None:
            self.data = [0.0] * (rows * cols)
        else:
            flat = [float(v) for v in data]
            expected = rows * cols
            if len(flat) != expected:
                raise ValueError(f"expected {expected} values, got {len(flat)}")
            self.data = flat

    # -- constructors -----------------------------------------------------
    @classmethod
    def from_nested(cls, nested) -> "Tensor":
        if isinstance(nested, Tensor):
            return nested.clone()
        if not isinstance(nested, (list, tuple)) or not nested:
            raise ValueError("tensor() expects a non-empty list")
        if isinstance(nested[0], Tensor):
            nested = [item.row(0) if item.cols == 1 else item.to_list() for item in nested]
        if not isinstance(nested[0], (list, tuple)):
            flat = [float(v) for v in nested]
            return cls(len(flat), 1, flat)
        rows = len(nested)
        cols = len(nested[0])
        data = []
        for row in nested:
            if len(row) != cols:
                raise ValueError("all rows must have the same length")
            data.extend(float(v) for v in row)
        return cls(rows, cols, data)

    @classmethod
    def zeros(cls, rows: int, cols: int) -> "Tensor":
        return cls(rows, cols)

    @classmethod
    def ones(cls, rows: int, cols: int) -> "Tensor":
        return cls(rows, cols, [1.0] * (rows * cols))

    @classmethod
    def randn(cls, rows: int, cols: int, rng: random.Random | None = None) -> "Tensor":
        rng = rng or random
        return cls(rows, cols, [rng.gauss(0.0, 1.0) for _ in range(rows * cols)])

    # -- access -----------------------------------------------------------
    @property
    def shape(self):
        return (self.rows, self.cols)

    def at(self, r: int, c: int) -> float:
        return self.data[r * self.cols + c]

    def set(self, r: int, c: int, value: float):
        self.data[r * self.cols + c] = value

    def row(self, index: int):
        start = index * self.cols
        return self.data[start:start + self.cols]

    def to_list(self):
        return [list(self.row(r)) for r in range(self.rows)]

    def column(self, index: int):
        return [self.data[r * self.cols + index] for r in range(self.rows)]

    def clone(self) -> "Tensor":
        return Tensor(self.rows, self.cols, list(self.data))

    # -- maths ------------------------------------------------------------
    def __matmul__(self, other: "Tensor") -> "Tensor":
        return matmul(self, other)

    def _elementwise(self, other, op):
        if isinstance(other, Tensor):
            if other.rows == 1 and other.cols == 1:
                scalar = other.data[0]
            elif other.rows == self.rows and other.cols == self.cols:
                return Tensor(self.rows, self.cols, [op(a, b) for a, b in zip(self.data, other.data)])
            else:
                raise ValueError(f"cannot combine shapes ({self.rows},{self.cols}) "
                                 f"and ({other.rows},{other.cols})")
        elif isinstance(other, (int, float)) and not isinstance(other, bool):
            scalar = float(other)
        else:
            return NotImplemented
        return Tensor(self.rows, self.cols, [op(value, scalar) for value in self.data])

    def __add__(self, other):
        return self._elementwise(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self._elementwise(other, lambda a, b: b + a)

    def __sub__(self, other):
        return self._elementwise(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._elementwise(other, lambda a, b: b - a)

    def __mul__(self, other):
        return self._elementwise(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self._elementwise(other, lambda a, b: b * a)

    def __truediv__(self, other):
        return self._elementwise(other, lambda a, b: a / b)

    def __neg__(self):
        return Tensor(self.rows, self.cols, [-value for value in self.data])

    def __len__(self):
        return self.rows

    def __repr__(self):  # pragma: no cover - debugging helper
        return f"Tensor(shape=({self.rows}, {self.cols}))"


def matmul(a: Tensor, b: Tensor) -> Tensor:
    if a.cols != b.rows:
        raise ValueError(f"cannot multiply shapes ({a.rows},{a.cols}) @ ({b.rows},{b.cols})")
    if _np is not None:
        arr = _np.asarray(a.data, dtype=float).reshape(a.rows, a.cols) @ \
              _np.asarray(b.data, dtype=float).reshape(b.rows, b.cols)
        return Tensor(a.rows, b.cols, arr.reshape(-1).tolist())
    out = [0.0] * (a.rows * b.cols)
    for i in range(a.rows):
        arow = a.row(i)
        for k in range(a.cols):
            av = arow[k]
            if av == 0.0:
                continue
            base = k * b.cols
            for j in range(b.cols):
                out[i * b.cols + j] += av * b.data[base + j]
    return Tensor(a.rows, b.cols, out)


def transpose(a: Tensor) -> Tensor:
    out = [0.0] * (a.rows * a.cols)
    for r in range(a.rows):
        for c in range(a.cols):
            out[c * a.rows + r] = a.at(r, c)
    return Tensor(a.cols, a.rows, out)


def softmax(values):
    values = [float(v) for v in values]
    if not values:
        return []
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


# --------------------------------------------------------------------------
# Activations
# --------------------------------------------------------------------------
def _sigmoid_scalar(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


ACTIVATIONS = {
    "none": lambda z: z,
    "linear": lambda z: z,
    "relu": lambda z: z if z > 0 else 0.0,
    "sigmoid": _sigmoid_scalar,
    "tanh": math.tanh,
    "gelu": lambda z: 0.5 * z * (1.0 + math.tanh(0.7978845608 * (z + 0.044715 * z ** 3))),
    "softmax": softmax,
}


def activation_forward(name: str, values):
    name = (name or "none").lower()
    if name == "softmax":
        # softmax works on the whole vector at once, not element by element
        return softmax(values)
    fn = ACTIVATIONS.get(name, ACTIVATIONS["none"])
    return [fn(v) for v in values]


def activation_grad(name: str, pre_values, post_values):
    """Derivative of the activation, element-wise."""
    name = name.lower()
    if name in ("none", "linear"):
        return [1.0] * len(pre_values)
    if name == "relu":
        return [1.0 if v > 0 else 0.0 for v in pre_values]
    if name == "sigmoid":
        return [p * (1.0 - p) for p in post_values]
    if name == "tanh":
        return [1.0 - p * p for p in post_values]
    if name == "gelu":
        out = []
        for z, p in zip(pre_values, post_values):
            c = 0.7978845608
            inner = c * (z + 0.044715 * z ** 3)
            d_inner = c * (1 + 3 * 0.044715 * z ** 2)
            out.append(0.5 * (1 + math.tanh(inner)) + 0.5 * p * (1 - math.tanh(inner) ** 2) * d_inner)
        return out
    if name == "softmax":
        # softmax is always used together with the crossentropy loss, which
        # already returns the fused gradient (probabilities - one hot)
        return [1.0] * len(pre_values)
    return [1.0] * len(pre_values)


# --------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------
class Layer:
    type_name = "Layer"
    trainable = False

    def build(self, in_dim: int, rng: random.Random):
        self.in_dim = in_dim

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError

    def params(self):
        return []

    def to_dict(self):
        return {"type": self.type_name}

    def load_state(self, state):
        pass


class Dense(Layer):
    type_name = "Dense"
    trainable = True

    def __init__(self, units: int, activation: str = "none"):
        self.units = int(units)
        self.activation = (activation or "none").lower()
        self.W = None
        self.b = None
        self.gW = None
        self.gb = None
        self._cache = None

    def build(self, in_dim: int, rng: random.Random):
        self.in_dim = in_dim
        limit = math.sqrt(6.0 / (in_dim + self.units))
        self.W = [[rng.uniform(-limit, limit) for _ in range(self.units)] for _ in range(in_dim)]
        self.b = [0.0] * self.units
        self.gW = [[0.0] * self.units for _ in range(in_dim)]
        self.gb = [0.0] * self.units

    def forward(self, x: Tensor) -> Tensor:
        if self.W is None:
            self.build(x.cols, random.Random(0))
        z = matmul(x, Tensor(self.in_dim, self.units, [v for row in self.W for v in row]))
        for r in range(z.rows):
            for j in range(self.units):
                z.set(r, j, z.at(r, j) + self.b[j])
        a = Tensor(z.rows, z.cols)
        for r in range(z.rows):
            activated = activation_forward(self.activation, z.row(r))
            for j in range(self.units):
                a.set(r, j, activated[j])
        self._cache = (x, z, a)
        return a

    def backward(self, grad: Tensor) -> Tensor:
        x, z, a = self._cache
        n = max(1, x.rows)
        dact = activation_grad(self.activation, z.data, a.data)
        dz = [grad.data[i] * dact[i] for i in range(len(grad.data))]
        # gradients
        for r in range(x.rows):
            xrow = x.row(r)
            for i in range(self.in_dim):
                xv = xrow[i]
                if xv == 0.0:
                    continue
                grow = self.gW[i]
                for j in range(self.units):
                    grow[j] += xv * dz[r * self.units + j]
        for j in range(self.units):
            total = 0.0
            for r in range(x.rows):
                total += dz[r * self.units + j]
            self.gb[j] += total
        dx = [0.0] * (x.rows * self.in_dim)
        for r in range(x.rows):
            for j in range(self.units):
                d = dz[r * self.units + j]
                if d == 0.0:
                    continue
                for i in range(self.in_dim):
                    dx[r * self.in_dim + i] += d * self.W[i][j]
        return Tensor(x.rows, self.in_dim, dx)

    def params(self):
        return [
            ("W", self.W, self.gW),
            ("b", self.b, self.gb),
        ]

    def zero_grad(self):
        self.gW = [[0.0] * self.units for _ in range(self.in_dim)]
        self.gb = [0.0] * self.units

    def to_dict(self):
        return {
            "type": "Dense",
            "units": self.units,
            "activation": self.activation,
            "W": self.W,
            "b": self.b,
        }

    def load_state(self, state):
        self.W = [[float(v) for v in row] for row in state.get("W", [])]
        self.b = [float(v) for v in state.get("b", [])]
        if self.W:
            self.in_dim = len(self.W)
            self.build_from_weights()

    def build_from_weights(self):
        self.gW = [[0.0] * self.units for _ in range(self.in_dim)]
        self.gb = [0.0] * self.units


class Activation(Layer):
    type_name = "Activation"

    def __init__(self, kind: str = "relu"):
        self.kind = (kind or "relu").lower()
        self._cache = None

    def forward(self, x: Tensor) -> Tensor:
        out = Tensor(x.rows, x.cols)
        for r in range(x.rows):
            activated = activation_forward(self.kind, x.row(r))
            for j in range(x.cols):
                out.set(r, j, activated[j])
        self._cache = (x, out)
        return out

    def backward(self, grad: Tensor) -> Tensor:
        x, a = self._cache
        d = activation_grad(self.kind, x.data, a.data)
        return Tensor(x.rows, x.cols, [grad.data[i] * d[i] for i in range(len(d))])

    def to_dict(self):
        return {"type": "Activation", "activation": self.kind}


class Dropout(Layer):
    type_name = "Dropout"

    def __init__(self, rate: float = 0.5):
        self.rate = float(rate)
        self.training = True
        self._mask = None

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.rate <= 0:
            self._mask = None
            return x
        keep = 1.0 - self.rate
        mask = [1.0 / keep if random.random() < keep else 0.0 for _ in range(x.rows * x.cols)]
        self._mask = mask
        return Tensor(x.rows, x.cols, [x.data[i] * mask[i] for i in range(len(mask))])

    def backward(self, grad: Tensor) -> Tensor:
        if self._mask is None:
            return grad
        return Tensor(grad.rows, grad.cols, [grad.data[i] * self._mask[i] for i in range(len(self._mask))])

    def to_dict(self):
        return {"type": "Dropout", "rate": self.rate}


class Flatten(Layer):
    type_name = "Flatten"

    def forward(self, x: Tensor) -> Tensor:
        return x.clone()

    def backward(self, grad: Tensor) -> Tensor:
        return grad

    def to_dict(self):
        return {"type": "Flatten"}


class Conv2d(Layer):
    """2-D convolution over small images.

    Images are stored flattened: one sample per row, pixels ordered as
    ``channel -> row -> column`` (``C * H * W`` values per sample).
    """

    type_name = "Conv2d"
    trainable = True

    def __init__(self, filters: int, kernel: int, height: int, width: int,
                 channels: int = 1, activation: str = "relu", stride: int = 1):
        self.filters = int(filters)
        self.kernel = int(kernel)
        self.height = int(height)
        self.width = int(width)
        self.channels = int(channels)
        self.activation = (activation or "none").lower()
        self.stride = max(1, int(stride))
        self.out_height = (self.height - self.kernel) // self.stride + 1
        self.out_width = (self.width - self.kernel) // self.stride + 1
        if self.out_height <= 0 or self.out_width <= 0:
            raise ValueError("the kernel is larger than the image")
        self.in_dim = self.channels * self.height * self.width
        self.out_dim = self.filters * self.out_height * self.out_width
        self.W = None
        self.b = None
        self.gW = None
        self.gb = None
        self._cache = None

    def build(self, in_dim: int, rng: random.Random):
        if in_dim and in_dim != self.in_dim:
            raise ValueError(f"Conv2d expects {self.in_dim} inputs per image, got {in_dim}")
        self.in_dim = self.in_dim or in_dim
        limit = math.sqrt(6.0 / (self.in_dim + self.out_dim))
        size = self.channels * self.kernel * self.kernel
        self.W = [[rng.uniform(-limit, limit) for _ in range(size)] for _ in range(self.filters)]
        self.b = [0.0] * self.filters
        self.gW = [[0.0] * size for _ in range(self.filters)]
        self.gb = [0.0] * self.filters

    def forward(self, x: Tensor) -> Tensor:
        if self.W is None:
            self.build(x.cols, random.Random(0))
        batch = x.rows
        data = x.data
        out = [0.0] * (batch * self.out_dim)
        pre = [0.0] * (batch * self.out_dim)
        k = self.kernel
        for s in range(batch):
            xoff = s * self.in_dim
            for f in range(self.filters):
                weights = self.W[f]
                bias = self.b[f]
                for oh in range(self.out_height):
                    for ow in range(self.out_width):
                        total = bias
                        for c in range(self.channels):
                            base = xoff + c * self.height * self.width
                            for i in range(k):
                                h = oh * self.stride + i
                                row_base = base + h * self.width
                                wbase = c * k * k + i * k
                                for j in range(k):
                                    wpos = ow * self.stride + j
                                    total += data[row_base + wpos] * weights[wbase + j]
                        index = s * self.out_dim + (f * self.out_height + oh) * self.out_width + ow
                        pre[index] = total
        activated = [0.0] * len(pre)
        for s in range(batch):
            for f in range(self.filters):
                start = s * self.out_dim + f * self.out_height * self.out_width
                block = pre[start:start + self.out_height * self.out_width]
                values = activation_forward(self.activation, block)
                activated[start:start + len(values)] = values
        self._cache = (x, pre, activated)
        return Tensor(batch, self.out_dim, activated)

    def backward(self, grad: Tensor) -> Tensor:
        x, pre, activated = self._cache
        batch = x.rows
        data = x.data
        k = self.kernel
        dx = [0.0] * (batch * self.in_dim)
        dpre = [0.0] * (batch * self.out_dim)
        for s in range(batch):
            start = s * self.out_dim
            block = pre[start:start + self.out_dim]
            post = activated[start:start + self.out_dim]
            derivative = activation_grad(self.activation, block, post)
            for i in range(len(derivative)):
                dpre[start + i] = grad.data[start + i] * derivative[i]
        for s in range(batch):
            xoff = s * self.in_dim
            for f in range(self.filters):
                weights = self.W[f]
                grow = self.gW[f]
                for oh in range(self.out_height):
                    for ow in range(self.out_width):
                        index = s * self.out_dim + (f * self.out_height + oh) * self.out_width + ow
                        grad_value = dpre[index]
                        if grad_value == 0.0:
                            continue
                        self.gb[f] += grad_value
                        for c in range(self.channels):
                            base = xoff + c * self.height * self.width
                            for i in range(k):
                                h = oh * self.stride + i
                                row_base = base + h * self.width
                                wbase = c * k * k + i * k
                                for j in range(k):
                                    wpos = ow * self.stride + j
                                    dx[row_base + wpos] += grad_value * weights[wbase + j]
                                    grow[wbase + j] += data[row_base + wpos] * grad_value
        return Tensor(batch, self.in_dim, dx)

    def params(self):
        return [("W", self.W, self.gW), ("b", self.b, self.gb)]

    def zero_grad(self):
        size = self.channels * self.kernel * self.kernel
        self.gW = [[0.0] * size for _ in range(self.filters)]
        self.gb = [0.0] * self.filters

    def to_dict(self):
        return {
            "type": "Conv2d", "filters": self.filters, "kernel": self.kernel,
            "height": self.height, "width": self.width, "channels": self.channels,
            "activation": self.activation, "stride": self.stride,
            "W": self.W, "b": self.b,
        }

    def load_state(self, state):
        self.W = [[float(v) for v in row] for row in state.get("W", [])]
        self.b = [float(v) for v in state.get("b", [])]
        if self.W:
            self.filters = len(self.W)
            self.zero_grad()


class MaxPool(Layer):
    """Max pooling over small images (keeps the channel layout)."""

    type_name = "MaxPool"

    def __init__(self, size: int = 2, stride: int = 2, height: int = 0, width: int = 0,
                 channels: int = 1):
        self.size = max(1, int(size))
        self.stride = max(1, int(stride))
        self.height = int(height)
        self.width = int(width)
        self.channels = int(channels)
        self.in_dim = self.channels * self.height * self.width if self.height else 0
        self.out_dim = 0
        self._cache = None

    def build(self, in_dim: int, rng: random.Random):
        if self.height:
            self.in_dim = self.in_dim or in_dim
        else:
            raise ValueError("MaxPool needs the image height and width")
        self.out_height = (self.height - self.size) // self.stride + 1
        self.out_width = (self.width - self.size) // self.stride + 1
        self.out_dim = self.channels * self.out_height * self.out_width

    def forward(self, x: Tensor) -> Tensor:
        batch = x.rows
        data = x.data
        out = [0.0] * (batch * self.out_dim)
        argmax = [0] * (batch * self.out_dim)
        for s in range(batch):
            xoff = s * self.in_dim
            for c in range(self.channels):
                base = xoff + c * self.height * self.width
                for oh in range(self.out_height):
                    for ow in range(self.out_width):
                        best = None
                        best_index = 0
                        for i in range(self.size):
                            h = oh * self.stride + i
                            row_base = base + h * self.width
                            for j in range(self.size):
                                wpos = ow * self.stride + j
                                value = data[row_base + wpos]
                                if best is None or value > best:
                                    best = value
                                    best_index = row_base + wpos
                        index = s * self.out_dim + (c * self.out_height + oh) * self.out_width + ow
                        out[index] = best if best is not None else 0.0
                        argmax[index] = best_index
        self._cache = (x, argmax)
        return Tensor(batch, self.out_dim, out)

    def backward(self, grad: Tensor) -> Tensor:
        x, argmax = self._cache
        dx = [0.0] * (x.rows * self.in_dim)
        for index in range(len(grad.data)):
            dx[argmax[index]] += grad.data[index]
        return Tensor(x.rows, self.in_dim, dx)

    def to_dict(self):
        return {"type": "MaxPool", "size": self.size, "stride": self.stride,
                "height": self.height, "width": self.width, "channels": self.channels}


class Embedding(Layer):
    type_name = "Embedding"
    trainable = True

    def __init__(self, vocab_size: int, dim: int):
        self.vocab_size = int(vocab_size)
        self.dim = int(dim)
        self.table = None
        self.grad = None
        self._indices = None

    def build(self, in_dim: int, rng: random.Random):
        self.table = [[rng.uniform(-0.5 / math.sqrt(self.dim), 0.5 / math.sqrt(self.dim))
                       for _ in range(self.dim)] for _ in range(self.vocab_size)]
        self.grad = [[0.0] * self.dim for _ in range(self.vocab_size)]

    def forward(self, x: Tensor) -> Tensor:
        if self.table is None:
            self.build(self.vocab_size, random.Random(0))
        indices = [int(v) for v in x.data]
        self._indices = indices
        out = Tensor(len(indices), self.dim)
        for r, idx in enumerate(indices):
            idx = max(0, min(self.vocab_size - 1, idx))
            for j in range(self.dim):
                out.set(r, j, self.table[idx][j])
        return out

    def backward(self, grad: Tensor) -> Tensor:
        for r, idx in enumerate(self._indices or []):
            idx = max(0, min(self.vocab_size - 1, idx))
            for j in range(self.dim):
                self.grad[idx][j] += grad.at(r, j)
        return Tensor(len(self._indices or []), 1)

    def zero_grad(self):
        self.grad = [[0.0] * self.dim for _ in range(self.vocab_size)]

    def params(self):
        return [("table", self.table, self.grad)]

    def to_dict(self):
        return {"type": "Embedding", "vocab_size": self.vocab_size, "dim": self.dim, "table": self.table}

    def load_state(self, state):
        self.table = [[float(v) for v in row] for row in state.get("table", [])]
        self.vocab_size = len(self.table)
        self.dim = len(self.table[0]) if self.table else self.dim
        self.zero_grad()


LAYER_TYPES = {
    "dense": Dense,
    "activation": Activation,
    "dropout": Dropout,
    "flatten": Flatten,
    "embedding": Embedding,
    "conv2d": Conv2d,
    "conv": Conv2d,
    "maxpool": MaxPool,
    "max_pool": MaxPool,
    "pool": MaxPool,
}


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------
class Loss:
    name = "mse"

    def __call__(self, yhat: Tensor, y):
        raise NotImplementedError

    @staticmethod
    def _as_vector(y, cols: int):
        """Return (flat list, is_one_hot_matrix)."""
        if isinstance(y, Tensor):
            if y.rows == 1 and y.cols == cols:
                return list(y.data), False
            return [y.data[r * y.cols] for r in range(y.rows)], False
        values = list(y)
        if values and isinstance(values[0], (list, tuple)):
            flat = []
            for row in values:
                flat.extend(float(v) for v in row)
            return flat, True
        return [float(v) for v in values], False


class MSELoss(Loss):
    name = "mse"

    def __call__(self, yhat: Tensor, y):
        target, _ = self._as_vector(y, yhat.cols)
        n = max(1, len(target))
        diffs = []
        total = 0.0
        for i in range(len(target)):
            diff = yhat.data[i] - target[i]
            diffs.append(diff)
            total += diff * diff
        loss = total / n
        grad = [2.0 * d / n for d in diffs]
        return loss, Tensor(yhat.rows, yhat.cols, grad)


class MAELoss(Loss):
    name = "mae"

    def __call__(self, yhat: Tensor, y):
        target, _ = self._as_vector(y, yhat.cols)
        n = max(1, len(target))
        total = 0.0
        grad = []
        for i in range(len(target)):
            diff = yhat.data[i] - target[i]
            total += abs(diff)
            grad.append(math.copysign(1.0 / n, diff) if diff != 0 else 0.0)
        return total / n, Tensor(yhat.rows, yhat.cols, grad)


class BinaryCrossEntropyLoss(Loss):
    name = "binary_crossentropy"

    def __call__(self, yhat: Tensor, y):
        target, _ = self._as_vector(y, yhat.cols)
        n = max(1, len(target))
        eps = 1e-7
        total = 0.0
        grad = []
        for i in range(len(target)):
            p = min(max(yhat.data[i], eps), 1 - eps)
            t = target[i]
            total += -(t * math.log(p) + (1 - t) * math.log(1 - p))
            grad.append((p - t) / (p * (1 - p) + eps) / n)
        return total / n, Tensor(yhat.rows, yhat.cols, grad)


class CrossEntropyLoss(Loss):
    """Softmax + cross entropy. Targets are class indices."""

    name = "crossentropy"

    def __call__(self, yhat: Tensor, y):
        n = yhat.rows
        indices = [int(v) for v in y]
        total = 0.0
        grad = [0.0] * (n * yhat.cols)
        for r in range(n):
            probs = softmax(yhat.row(r))
            label = indices[r] if r < len(indices) else 0
            label = max(0, min(yhat.cols - 1, label))
            total += -math.log(max(probs[label], 1e-12))
            for c in range(yhat.cols):
                grad[r * yhat.cols + c] = probs[c] - (1.0 if c == label else 0.0)
        return total / max(1, n), Tensor(n, yhat.cols, [g / max(1, n) for g in grad])


LOSSES = {
    "mse": MSELoss,
    "mean_squared_error": MSELoss,
    "mae": MAELoss,
    "binary_crossentropy": BinaryCrossEntropyLoss,
    "bce": BinaryCrossEntropyLoss,
    "crossentropy": CrossEntropyLoss,
    "cross_entropy": CrossEntropyLoss,
}


# --------------------------------------------------------------------------
# Optimizers
# --------------------------------------------------------------------------
class Optimizer:
    """Base optimizer.

    Every parameter keeps its own state, so a 2-D weight matrix and a 1-D
    bias vector are updated independently and element by element.
    """

    def __init__(self, name: str = "sgd", lr: float = 0.01):
        self.name = (name or "sgd").lower()
        self.lr = float(lr)
        self.step_count = 0
        self.state = {}

    @staticmethod
    def _zeros_like(values):
        if isinstance(values[0], list):
            return [[0.0] * len(row) for row in values]
        return [0.0] * len(values)

    def _slots(self, index, values):
        if index not in self.state:
            self.state[index] = self._zeros_like(values)
        return self.state[index]

    def step(self, params):
        self.step_count += 1
        for index, (_name, values, grads) in enumerate(params):
            self._step(index, values, grads)

    def _step(self, index, values, grads):
        if isinstance(values[0], list):
            for i in range(len(values)):
                row = values[i]
                grow = grads[i]
                for j in range(len(row)):
                    row[j] -= self.lr * grow[j]
        else:
            for i in range(len(values)):
                values[i] -= self.lr * grads[i]


class Momentum(Optimizer):
    def __init__(self, name="momentum", lr=0.01, mu=0.9):
        super().__init__(name, lr)
        self.mu = mu

    def _step(self, index, values, grads):
        velocity = self._slots(index, values)
        if isinstance(values[0], list):
            for i in range(len(values)):
                for j in range(len(values[i])):
                    velocity[i][j] = self.mu * velocity[i][j] + grads[i][j]
                    values[i][j] -= self.lr * velocity[i][j]
        else:
            for i in range(len(values)):
                velocity[i] = self.mu * velocity[i] + grads[i]
                values[i] -= self.lr * velocity[i]


class RMSProp(Optimizer):
    def __init__(self, name="rmsprop", lr=0.01, alpha=0.9, eps=1e-8):
        super().__init__(name, lr)
        self.alpha = alpha
        self.eps = eps

    def _step(self, index, values, grads):
        square = self._slots(index, values)
        if isinstance(values[0], list):
            for i in range(len(values)):
                for j in range(len(values[i])):
                    square[i][j] = self.alpha * square[i][j] + (1 - self.alpha) * grads[i][j] ** 2
                    values[i][j] -= self.lr * grads[i][j] / (math.sqrt(square[i][j]) + self.eps)
        else:
            for i in range(len(values)):
                square[i] = self.alpha * square[i] + (1 - self.alpha) * grads[i] ** 2
                values[i] -= self.lr * grads[i] / (math.sqrt(square[i]) + self.eps)


class Adam(Optimizer):
    def __init__(self, name="adam", lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(name, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def _step(self, index, values, grads):
        if index not in self.state:
            self.state[index] = (self._zeros_like(values), self._zeros_like(values))
        moment, second = self.state[index]
        t = self.step_count
        correction1 = 1 - self.beta1 ** t
        correction2 = 1 - self.beta2 ** t
        if isinstance(values[0], list):
            for i in range(len(values)):
                for j in range(len(values[i])):
                    grad = grads[i][j]
                    moment[i][j] = self.beta1 * moment[i][j] + (1 - self.beta1) * grad
                    second[i][j] = self.beta2 * second[i][j] + (1 - self.beta2) * grad * grad
                    m_hat = moment[i][j] / correction1
                    v_hat = second[i][j] / correction2
                    values[i][j] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
        else:
            for i in range(len(values)):
                grad = grads[i]
                moment[i] = self.beta1 * moment[i] + (1 - self.beta1) * grad
                second[i] = self.beta2 * second[i] + (1 - self.beta2) * grad * grad
                m_hat = moment[i] / correction1
                v_hat = second[i] / correction2
                values[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)


OPTIMIZERS = {
    "sgd": Optimizer,
    "momentum": Momentum,
    "rmsprop": RMSProp,
    "adam": Adam,
}


# --------------------------------------------------------------------------
# Dataset & model
# --------------------------------------------------------------------------
class Dataset:
    def __init__(self, x: Tensor, y, name: str = "dataset"):
        self.x = x
        self.y = y
        self.name = name

    def __len__(self):
        return self.x.rows

    @property
    def input_dim(self):
        return self.x.cols

    @property
    def classes(self):
        if isinstance(self.y, Tensor):
            return self.y.cols
        return len(set(int(v) for v in self.y)) or 1

    def slice(self, start: int, size: int):
        end = min(self.x.rows, start + size)
        x = Tensor(end - start, self.x.cols, self.x.data[start * self.x.cols:end * self.x.cols])
        if isinstance(self.y, Tensor):
            y = Tensor(end - start, self.y.cols, self.y.data[start * self.y.cols:end * self.y.cols])
        else:
            y = list(self.y[start:end])
        return Dataset(x, y, self.name)

    def describe(self):
        return {
            "name": self.name,
            "samples": self.x.rows,
            "features": self.x.cols,
            "classes": self.classes,
        }


class Model:
    def __init__(self, name: str = "model"):
        self.name = name
        self.layers: list[Layer] = []
        self.history: list[dict] = []
        self.metrics: dict = {}

    # -- structure --------------------------------------------------------
    def add_layer(self, layer: Layer):
        self.layers.append(layer)
        return layer

    def layer_names(self):
        return [layer.type_name for layer in self.layers]

    def build(self, rng: random.Random, input_dim: int | None = None):
        dim = input_dim
        spatial = None  # (height, width, channels) of the last feature map
        for layer in self.layers:
            if isinstance(layer, Embedding):
                layer.build(layer.vocab_size, rng)
                dim = layer.dim
                continue
            if isinstance(layer, Conv2d):
                layer.build(dim or layer.in_dim, rng)
                dim = layer.out_dim
                spatial = (layer.out_height, layer.out_width, layer.filters)
                continue
            if isinstance(layer, MaxPool):
                if not layer.height:
                    if spatial is None:
                        raise ValueError("MaxPool needs the image size: put a Conv2d before it")
                    layer.height, layer.width, layer.channels = spatial
                    layer.in_dim = layer.channels * layer.height * layer.width
                layer.build(layer.in_dim, rng)
                dim = layer.out_dim
                spatial = (layer.out_height, layer.out_width, layer.channels)
                continue
            if dim is None:
                dim = getattr(layer, "in_dim", 0) or 0
            layer.build(dim, rng)
            if isinstance(layer, Dense):
                dim = layer.units
            elif getattr(layer, "out_dim", None):
                dim = layer.out_dim

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad: Tensor) -> Tensor:
        out = grad
        for layer in reversed(self.layers):
            out = layer.backward(out)
        return out

    def params(self):
        collected = []
        for layer in self.layers:
            collected.extend(layer.params())
        return collected

    def zero_grad(self):
        for layer in self.layers:
            if hasattr(layer, "zero_grad"):
                layer.zero_grad()

    # -- inference --------------------------------------------------------
    def predict(self, x: Tensor):
        out = self.forward(x)
        if out.cols == 1:
            return [1 if v > 0.5 else 0 for v in out.data]
        return [max(range(out.cols), key=lambda c: out.at(r, c)) for r in range(out.rows)]

    def predict_proba(self, x: Tensor):
        out = self.forward(x)
        if out.cols == 1:
            return [[min(max(v, 0.0), 1.0)] for v in out.data]
        return [softmax(out.row(r)) for r in range(out.rows)]

    def evaluate(self, dataset: Dataset):
        preds = self.predict(dataset.x)
        truth = dataset.y
        if isinstance(truth, Tensor):
            truth = [int(truth.data[r]) for r in range(truth.rows)]
        correct = sum(1 for p, t in zip(preds, truth) if p == int(t))
        accuracy = correct / max(1, len(truth))
        result = {"accuracy": accuracy, "correct": correct, "total": len(truth)}
        result.update(classification_metrics(preds, truth))
        self.metrics = result
        return result

    # -- serialisation ----------------------------------------------------
    def to_dict(self):
        return {
            "format": "lume-model/1",
            "name": self.name,
            "layers": [layer.to_dict() for layer in self.layers],
            "metrics": self.metrics,
            "history": self.history[-10:],
        }

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
        return path

    @staticmethod
    def load(path: str) -> "Model":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        model = Model(data.get("name", "model"))
        for state in data.get("layers", []):
            kind = str(state.get("type", "")).lower()
            if kind == "dense":
                layer = Dense(state.get("units", 1), state.get("activation", "none"))
            elif kind == "activation":
                layer = Activation(state.get("activation", "relu"))
            elif kind == "dropout":
                layer = Dropout(state.get("rate", 0.5))
            elif kind == "flatten":
                layer = Flatten()
            elif kind == "embedding":
                layer = Embedding(state.get("vocab_size", 1), state.get("dim", 1))
            else:
                continue
            layer.load_state(state)
            model.add_layer(layer)
        model.metrics = data.get("metrics", {})
        return model


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def classification_metrics(preds, truth) -> dict:
    classes = sorted(set(int(t) for t in truth) | set(int(p) for p in preds))
    precisions, recalls, f1s = [], [], []
    for c in classes:
        tp = sum(1 for p, t in zip(preds, truth) if p == c and int(t) == c)
        fp = sum(1 for p, t in zip(preds, truth) if p == c and int(t) != c)
        fn = sum(1 for p, t in zip(preds, truth) if p != c and int(t) == c)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    count = len(classes) or 1
    return {
        "precision": sum(precisions) / count,
        "recall": sum(recalls) / count,
        "f1": sum(f1s) / count,
        "confusion_matrix": confusion_matrix(preds, truth, classes),
    }


def confusion_matrix(preds, truth, classes=None):
    classes = classes or sorted(set(int(t) for t in truth) | set(int(p) for p in preds))
    index = {c: i for i, c in enumerate(classes)}
    matrix = [[0 for _ in classes] for _ in classes]
    for p, t in zip(preds, truth):
        matrix[index.get(int(t), 0)][index.get(int(p), 0)] += 1
    return matrix


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------
def train_model(model: Model, dataset: Dataset, epochs: int = 10, lr: float = 0.01,
                optimizer: str = "sgd", loss: str = "mse", batch_size: int = 16,
                verbose: bool = True, emit=None) -> list[dict]:
    """Train ``model`` on ``dataset`` and return the per-epoch history."""
    emit = emit or (lambda text: None)
    rng = random.Random(7)
    if not any(isinstance(layer, Embedding) for layer in model.layers):
        model.build(rng, dataset.x.cols)
    else:
        model.build(rng, dataset.x.cols)
    opt_cls = OPTIMIZERS.get(str(optimizer).lower(), Optimizer)
    opt = opt_cls(lr=lr)
    loss_cls = LOSSES.get(str(loss).lower(), MSELoss)
    loss_fn = loss_cls()
    n = len(dataset)
    if n == 0:
        raise ValueError("cannot train on an empty dataset")
    batch_size = max(1, min(int(batch_size), n))
    history = []
    for epoch in range(1, int(epochs) + 1):
        model.zero_grad()
        total_loss = 0.0
        batches = 0
        for start in range(0, n, batch_size):
            batch = dataset.slice(start, batch_size)
            out = model.forward(batch.x)
            value, grad = loss_fn(out, batch.y)
            total_loss += value
            batches += 1
            model.zero_grad()
            model.backward(grad)
            opt.step(model.params())
        avg_loss = total_loss / max(1, batches)
        metrics = model.evaluate(dataset)
        record = {"epoch": epoch, "loss": avg_loss, **metrics}
        history.append(record)
        model.history.append(record)
        if verbose:
            emit(f"epoch {epoch}/{epochs}  loss {avg_loss:.6f}  acc {metrics['accuracy']:.3f}  "
                 f"f1 {metrics.get('f1', 0):.3f}")
    return history
