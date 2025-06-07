# coding: utf-8
"""Deep LSTM-based agent.

This implementation provides a simple LSTM agent that learns
from past price data to predict future price movement. It
uses only ``numpy`` for the network to avoid external
dependencies.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .base import Agent
from ..logs.base import Logger
from ..market import Market
from ..order import LIMIT_ORDER, Cancel, Order


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def dsigmoid(y: np.ndarray) -> np.ndarray:
    return y * (1 - y)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def dtanh(y: np.ndarray) -> np.ndarray:
    return 1 - y ** 2


class SimpleLSTMNetwork:
    """A very small LSTM with one output layer."""

    def __init__(self, input_size: int = 2, hidden_size: int = 32) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        rng = np.random.default_rng(0)
        # weights for gates: concat(h,x) -> hidden
        self.Wf = rng.normal(scale=0.1, size=(hidden_size + input_size, hidden_size))
        self.bf = np.zeros(hidden_size)
        self.Wi = rng.normal(scale=0.1, size=(hidden_size + input_size, hidden_size))
        self.bi = np.zeros(hidden_size)
        self.Wo = rng.normal(scale=0.1, size=(hidden_size + input_size, hidden_size))
        self.bo = np.zeros(hidden_size)
        self.Wg = rng.normal(scale=0.1, size=(hidden_size + input_size, hidden_size))
        self.bg = np.zeros(hidden_size)
        # fully connected layers
        self.W1 = rng.normal(scale=0.1, size=(hidden_size, hidden_size))
        self.b1 = np.zeros(hidden_size)
        self.W2 = rng.normal(scale=0.1, size=(hidden_size, 2))
        self.b2 = np.zeros(2)
        self.lr = 0.001

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Any]:
        """Forward pass.

        Args:
            x: shape (seq_len, input_size)
        Returns:
            logits and cache for backprop
        """
        h = np.zeros((x.shape[0] + 1, self.hidden_size))
        c = np.zeros((x.shape[0] + 1, self.hidden_size))
        caches = []
        for t in range(x.shape[0]):
            inp = np.concatenate([h[t], x[t]])
            f = sigmoid(np.dot(inp, self.Wf) + self.bf)
            i = sigmoid(np.dot(inp, self.Wi) + self.bi)
            o = sigmoid(np.dot(inp, self.Wo) + self.bo)
            g = tanh(np.dot(inp, self.Wg) + self.bg)
            c[t + 1] = f * c[t] + i * g
            h[t + 1] = o * tanh(c[t + 1])
            caches.append((inp, f, i, o, g, c[t], c[t + 1]))
        out = h[-1]
        mu = out.mean()
        var = out.var() + 1e-6
        norm = (out - mu) / np.sqrt(var)
        pre = np.dot(norm, self.W1) + self.b1
        relu = np.maximum(pre, 0)
        logits = np.dot(relu, self.W2) + self.b2
        cache = (caches, h, c, norm, mu, var, pre, relu)
        return logits, cache

    def backward(self, x: np.ndarray, y: int, cache: Any) -> None:
        """Backward and update weights for one sample."""
        (caches, h, c, norm, mu, var, pre, relu) = cache
        probs = np.exp(pre * 0)  # dummy to ensure array creation
        # softmax
        ex = np.exp(np.dot(relu, self.W2) + self.b2 - np.max(np.dot(relu, self.W2) + self.b2))
        probs = ex / ex.sum()
        dlogits = probs
        dlogits[y] -= 1
        # fc2
        dW2 = np.outer(relu, dlogits)
        db2 = dlogits
        drelu = np.dot(dlogits, self.W2.T)
        dpre = drelu * (pre > 0)
        dW1 = np.outer(norm, dpre)
        db1 = dpre
        dnorm = np.dot(dpre, self.W1.T)
        dvar = (-0.5) * np.sum(dnorm * (h[-1] - mu)) / (var ** 1.5)
        dmu = -np.sum(dnorm) / np.sqrt(var) + dvar * (-2 * np.mean(h[-1] - mu))
        dh = dnorm / np.sqrt(var) + dvar * 2 * (h[-1] - mu) / h[-1].size + dmu / h[-1].size

        dh_next = dh
        dc_next = np.zeros(self.hidden_size)
        dWf = np.zeros_like(self.Wf)
        dWi = np.zeros_like(self.Wi)
        dWo = np.zeros_like(self.Wo)
        dWg = np.zeros_like(self.Wg)
        dbf = np.zeros_like(self.bf)
        dbi = np.zeros_like(self.bi)
        dbo = np.zeros_like(self.bo)
        dbg = np.zeros_like(self.bg)
        for t in reversed(range(x.shape[0])):
            inp, f, i, o, g, c_prev, c_curr = caches[t]
            dh_cur = dh_next
            do = dh_cur * tanh(c_curr) * dsigmoid(o)
            dc = dh_cur * o * dtanh(tanh(c_curr)) + dc_next
            df = dc * c_prev * dsigmoid(f)
            di = dc * g * dsigmoid(i)
            dg = dc * i * dtanh(g)
            dconcat = (
                np.dot(do, self.Wo.T)
                + np.dot(df, self.Wf.T)
                + np.dot(di, self.Wi.T)
                + np.dot(dg, self.Wg.T)
            )
            dh_next = dconcat[: self.hidden_size]
            dc_next = dc * f
            dxi = dconcat[self.hidden_size :]
            dWf += np.outer(inp, df)
            dWi += np.outer(inp, di)
            dWo += np.outer(inp, do)
            dWg += np.outer(inp, dg)
            dbf += df
            dbi += di
            dbo += do
            dbg += dg
        # update
        for param, grad in [
            (self.W2, dW2),
            (self.b2, db2),
            (self.W1, dW1),
            (self.b1, db1),
            (self.Wf, dWf),
            (self.bf, dbf),
            (self.Wi, dWi),
            (self.bi, dbi),
            (self.Wo, dWo),
            (self.bo, dbo),
            (self.Wg, dWg),
            (self.bg, dbg),
        ]:
            param -= self.lr * grad

    def predict(self, x: np.ndarray) -> int:
        logits, _ = self.forward(x)
        return int(np.argmax(logits))

    def train(self, X: List[np.ndarray], y: List[int], epochs: int = 1) -> None:
        for _ in range(epochs):
            for x_s, y_s in zip(X, y):
                logits, cache = self.forward(x_s)
                self.backward(x_s, y_s, cache)

    def accuracy(self, X: List[np.ndarray], y: List[int]) -> float:
        correct = 0
        for x_s, y_s in zip(X, y):
            pred = self.predict(x_s)
            if pred == y_s:
                correct += 1
        return correct / max(len(X), 1)


class DeepAgent(Agent):
    """Agent with an internal LSTM predictor."""

    sequence_length: int = 100
    prediction_horizon: int = 100

    def __init__(
        self,
        agent_id: int,
        prng: random.Random,
        simulator: "Simulator",  # type: ignore  # NOQA
        name: str,
        logger: Optional[Logger] = None,
    ) -> None:
        super().__init__(agent_id, prng, simulator, name, logger)
        self.net = SimpleLSTMNetwork()
        self.prices: List[float] = []
        self.mids: List[float] = []

    def setup(
        self,
        settings: Dict[str, Any],
        accessible_markets_ids: List[int],
        *args: Any,
        **kwargs: Any,
    ) -> None:  # type: ignore
        super().setup(settings=settings, accessible_markets_ids=accessible_markets_ids)
        if len(accessible_markets_ids) != 1:
            raise ValueError("DeepAgent currently supports one market")
        self.market_id = accessible_markets_ids[0]

    def _collect_data(self, market: Market) -> None:
        self.prices.append(market.get_market_price())
        self.mids.append(market.get_mid_price() or market.get_market_price())

    def _prepare_dataset(self) -> Tuple[List[np.ndarray], List[int]]:
        X: List[np.ndarray] = []
        y: List[int] = []
        seq = self.sequence_length
        horizon = self.prediction_horizon
        for t in range(len(self.mids) - seq - horizon):
            price_window = self.prices[t : t + seq]
            mid_window = self.mids[t : t + seq]
            arr = np.stack([price_window, mid_window], axis=1)
            future = self.mids[t + seq + horizon - 1]
            now = self.mids[t + seq - 1]
            label = 1 if future > now else 0
            X.append(arr.astype(float))
            y.append(label)
        return X, y

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        orders: List[Union[Order, Cancel]] = []
        market = markets[0]
        self._collect_data(market)
        if len(self.mids) <= self.sequence_length + self.prediction_horizon + 100:
            return orders
        X, y = self._prepare_dataset()
        if len(X) <= 100:
            return orders
        train_X = X[:-100]
        train_y = y[:-100]
        test_X = X[-100:]
        test_y = y[-100:]
        if train_X:
            self.net.train(train_X, train_y, epochs=1)
        acc = self.net.accuracy(test_X, test_y)
        if acc <= 0.51:
            return orders
        latest = np.stack(
            [self.prices[-self.sequence_length :], self.mids[-self.sequence_length :]],
            axis=1,
        ).astype(float)
        pred = self.net.predict(latest)
        volume = self.asset_volumes[self.market_id]
        if pred == 1 and volume < 1:
            orders.append(
                Order(
                    agent_id=self.agent_id,
                    market_id=self.market_id,
                    is_buy=True,
                    kind=LIMIT_ORDER,
                    volume=1,
                    price=market.get_market_price(),
                    ttl=1,
                )
            )
        if pred == 0 and volume > -1:
            orders.append(
                Order(
                    agent_id=self.agent_id,
                    market_id=self.market_id,
                    is_buy=False,
                    kind=LIMIT_ORDER,
                    volume=1,
                    price=market.get_market_price(),
                    ttl=1,
                )
            )
        return orders
