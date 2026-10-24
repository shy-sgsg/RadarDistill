"""Small NumPy 1D CNN for circular Doppler regression (no deep-learning runtime needed)."""

from __future__ import annotations

import numpy as np


def frequency_to_circle(frequency_hz: np.ndarray | float, prf_hz: float) -> np.ndarray:
    angle = 2.0 * np.pi * np.asarray(frequency_hz, dtype=np.float64) / prf_hz
    return np.stack((np.cos(angle), np.sin(angle)), axis=-1)


def circle_to_frequency(output: np.ndarray, prf_hz: float) -> np.ndarray | float:
    values = np.asarray(output)
    angle = np.arctan2(values[..., 1], values[..., 0])
    frequency = (angle * prf_hz / (2.0 * np.pi) + prf_hz / 2.0) % prf_hz - prf_hz / 2.0
    return float(frequency) if frequency.ndim == 0 else frequency


def _complex_features(iq: np.ndarray) -> np.ndarray:
    values = np.asarray(iq)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or not np.iscomplexobj(values):
        raise ValueError("iq must be a complex array with shape (N,) or (batch, N)")
    return np.stack((values.real, values.imag), axis=1).astype(np.float32, copy=False)


class TinyCircularCNN:
    """Conv1D front end with mean-pooled or position-preserving circular readout."""

    def __init__(
        self,
        n_samples: int = 130,
        seed: int = 0,
        channels: int = 6,
        kernel_size: int = 9,
        pooling: str = "mean",
        hidden_units: int = 32,
    ):
        if n_samples < kernel_size or channels <= 0 or kernel_size <= 0:
            raise ValueError("n_samples, channels, and kernel_size must be positive and compatible")
        if pooling not in {"mean", "flatten"}:
            raise ValueError("pooling must be 'mean' or 'flatten'")
        if pooling == "flatten" and hidden_units <= 0:
            raise ValueError("hidden_units must be positive for flatten pooling")
        self.n_samples = int(n_samples)
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.pooling = pooling
        self.hidden_units = int(hidden_units)
        rng = np.random.default_rng(seed)
        self.conv_weight = rng.normal(
            0.0, np.sqrt(2.0 / (2 * kernel_size)), size=(channels, 2, kernel_size)
        ).astype(np.float32)
        self.conv_bias = np.zeros(channels, dtype=np.float32)
        if pooling == "flatten":
            flattened_size = channels * (n_samples - kernel_size + 1)
            self.hidden_weight = rng.normal(
                0.0, np.sqrt(2.0 / flattened_size), size=(hidden_units, flattened_size)
            ).astype(np.float32)
            self.hidden_bias = np.zeros(hidden_units, dtype=np.float32)
            head_input_size = hidden_units
        else:
            self.hidden_weight = None
            self.hidden_bias = None
            head_input_size = channels
        self.head_weight = rng.normal(
            0.0, np.sqrt(1.0 / head_input_size), size=(2, head_input_size)
        ).astype(np.float32)
        self.head_bias = np.zeros(2, dtype=np.float32)

    def _forward(self, features: np.ndarray):
        windows = np.lib.stride_tricks.sliding_window_view(features, self.kernel_size, axis=-1)
        convolution = np.einsum("bclk,ock->bol", windows, self.conv_weight, optimize=True)
        convolution += self.conv_bias[None, :, None]
        activation = np.maximum(convolution, 0.0)
        if self.pooling == "mean":
            flattened = None
            hidden_preactivation = None
            head_input = np.mean(activation, axis=-1)
        else:
            flattened = activation.reshape(len(activation), -1)
            hidden_preactivation = flattened @ self.hidden_weight.T + self.hidden_bias
            head_input = np.maximum(hidden_preactivation, 0.0)
        output = head_input @ self.head_weight.T + self.head_bias
        return output, (windows, convolution, activation, head_input, flattened, hidden_preactivation)

    def predict(self, iq: np.ndarray) -> np.ndarray:
        features = _complex_features(iq)
        if features.shape[-1] != self.n_samples:
            raise ValueError(f"expected {self.n_samples} slow-time samples")
        output, _ = self._forward(features)
        return output

    def loss_value(
        self,
        iq: np.ndarray,
        targets: np.ndarray,
        teacher_targets: np.ndarray | None = None,
        distillation_weight: float = 0.0,
    ) -> float:
        """Evaluate the exact supervised plus optional distillation objective."""
        features = _complex_features(iq)
        y = np.asarray(targets, dtype=np.float32)
        teacher = None if teacher_targets is None else np.asarray(teacher_targets, dtype=np.float32)
        if features.shape[-1] != self.n_samples or y.shape != (len(features), 2):
            raise ValueError("input or target shape does not match the model")
        if teacher is not None and teacher.shape != y.shape:
            raise ValueError("teacher target shape must match the physical target")
        if distillation_weight < 0 or (teacher is None and distillation_weight != 0):
            raise ValueError("a nonnegative weight and matching teacher target are required")
        output, _ = self._forward(features)
        loss = float(np.mean((output - y) ** 2))
        if teacher is not None:
            loss += distillation_weight * float(np.mean((output - teacher) ** 2))
        return loss

    def loss_and_gradients(
        self,
        iq: np.ndarray,
        targets: np.ndarray,
        teacher_targets: np.ndarray | None = None,
        distillation_weight: float = 0.0,
    ) -> tuple[float, dict[str, np.ndarray]]:
        """Evaluate the training objective and return analytic parameter gradients."""
        features = _complex_features(iq)
        y = np.asarray(targets, dtype=np.float32)
        teacher = None if teacher_targets is None else np.asarray(teacher_targets, dtype=np.float32)
        if features.shape[-1] != self.n_samples or y.shape != (len(features), 2):
            raise ValueError("input or target shape does not match the model")
        if teacher is not None and teacher.shape != y.shape:
            raise ValueError("teacher target shape must match the physical target")
        if distillation_weight < 0 or (teacher is None and distillation_weight != 0):
            raise ValueError("a nonnegative weight and matching teacher target are required")
        return self._loss_and_gradients_from_features(features, y, teacher, distillation_weight)

    def _loss_and_gradients_from_features(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        teacher_targets: np.ndarray | None,
        distillation_weight: float,
    ) -> tuple[float, dict[str, np.ndarray]]:
        output, cache = self._forward(features)
        windows, convolution, activation, head_input, flattened, hidden_preactivation = cache
        grad_output = (output - targets) / len(features)
        loss = float(np.mean((output - targets) ** 2))
        if teacher_targets is not None:
            difference = output - teacher_targets
            grad_output += distillation_weight * difference / len(features)
            loss += distillation_weight * float(np.mean(difference**2))

        grad_head_weight = grad_output.T @ head_input
        grad_head_bias = np.sum(grad_output, axis=0)
        grad_head_input = grad_output @ self.head_weight
        gradients: dict[str, np.ndarray] = {
            "head_weight": grad_head_weight,
            "head_bias": grad_head_bias,
        }
        if self.pooling == "mean":
            grad_activation = np.broadcast_to(
                grad_head_input[:, :, None] / convolution.shape[-1], convolution.shape
            )
        else:
            grad_hidden = grad_head_input * (hidden_preactivation > 0.0)
            gradients["hidden_weight"] = grad_hidden.T @ flattened
            gradients["hidden_bias"] = np.sum(grad_hidden, axis=0)
            grad_flattened = grad_hidden @ self.hidden_weight
            grad_activation = grad_flattened.reshape(activation.shape)

        grad_convolution = grad_activation * (convolution > 0.0)
        gradients["conv_weight"] = np.einsum(
            "bol,bclk->ock", grad_convolution, windows, optimize=True
        )
        gradients["conv_bias"] = np.sum(grad_convolution, axis=(0, 2))
        return loss, gradients

    def fit(
        self,
        iq: np.ndarray,
        targets: np.ndarray,
        epochs: int = 50,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        seed: int = 0,
        teacher_targets: np.ndarray | None = None,
        distillation_weight: float = 0.0,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        patience: int | None = None,
    ) -> list[float]:
        """Train with circular MSE and optional teacher loss; return epoch losses."""
        features = _complex_features(iq)
        y = np.asarray(targets, dtype=np.float32)
        if features.shape[-1] != self.n_samples or y.shape != (len(features), 2):
            raise ValueError("input or target shape does not match the model")
        if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
            raise ValueError("epochs, batch_size, and learning_rate must be positive")
        teacher = None
        if teacher_targets is not None:
            teacher = np.asarray(teacher_targets, dtype=np.float32)
            if teacher.shape != y.shape or distillation_weight < 0:
                raise ValueError("teacher target shape must match, and weight must be nonnegative")
        elif distillation_weight != 0:
            raise ValueError("teacher_targets are required when distillation_weight is nonzero")

        params = [self.conv_weight, self.conv_bias]
        if self.pooling == "flatten":
            params.extend([self.hidden_weight, self.hidden_bias])
        params.extend([self.head_weight, self.head_bias])
        first_moments = [np.zeros_like(param) for param in params]
        second_moments = [np.zeros_like(param) for param in params]
        rng = np.random.default_rng(seed)
        history: list[float] = []
        best_state = None
        best_validation_loss = np.inf
        stale_epochs = 0
        step = 0

        if validation_data is not None:
            val_iq, val_targets = validation_data
            val_features = _complex_features(val_iq)
            val_y = np.asarray(val_targets, dtype=np.float32)
            if val_features.shape[-1] != self.n_samples or val_y.shape != (len(val_features), 2):
                raise ValueError("validation input or target shape does not match the model")
        else:
            val_features = val_y = None

        for _epoch in range(epochs):
            order = rng.permutation(len(features))
            epoch_loss = 0.0
            seen = 0
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                xb = features[indices]
                yb = y[indices]
                batch_loss, gradient_map = self._loss_and_gradients_from_features(
                    xb, yb, None if teacher is None else teacher[indices], distillation_weight
                )
                gradient_names = ["conv_weight", "conv_bias"]
                if self.pooling == "flatten":
                    gradient_names.extend(["hidden_weight", "hidden_bias"])
                gradient_names.extend(["head_weight", "head_bias"])
                gradients = [gradient_map[name] for name in gradient_names]

                step += 1
                for index, (parameter, gradient) in enumerate(zip(params, gradients)):
                    first_moments[index] = 0.9 * first_moments[index] + 0.1 * gradient
                    second_moments[index] = 0.999 * second_moments[index] + 0.001 * gradient**2
                    corrected_first = first_moments[index] / (1.0 - 0.9**step)
                    corrected_second = second_moments[index] / (1.0 - 0.999**step)
                    parameter -= learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
                epoch_loss += batch_loss * len(indices)
                seen += len(indices)
            history.append(epoch_loss / seen)

            if val_features is not None:
                val_prediction, _ = self._forward(val_features)
                validation_loss = float(np.mean((val_prediction - val_y) ** 2))
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_state = self.state_dict()
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if patience is not None and stale_epochs >= patience:
                        break
        if best_state is not None:
            self.load_state_dict(best_state)
        return history

    def state_dict(self) -> dict[str, np.ndarray]:
        state = {
            "conv_weight": self.conv_weight.copy(),
            "conv_bias": self.conv_bias.copy(),
            "head_weight": self.head_weight.copy(),
            "head_bias": self.head_bias.copy(),
        }
        if self.pooling == "flatten":
            state["hidden_weight"] = self.hidden_weight.copy()
            state["hidden_bias"] = self.hidden_bias.copy()
        return state

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        expected_names = set(self.state_dict())
        if set(state) != expected_names:
            raise ValueError("weights do not match this model architecture")
        for name in expected_names:
            value = np.asarray(state[name], dtype=np.float32)
            target = getattr(self, name)
            if value.shape != target.shape:
                raise ValueError(f"incorrect shape for {name}")
            target[...] = value

    def save(self, path: str) -> None:
        np.savez(path, **self.state_dict())

    def load(self, path: str) -> None:
        with np.load(path) as archive:
            self.load_state_dict({name: archive[name] for name in archive.files})
