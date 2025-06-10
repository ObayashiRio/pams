# coding: utf-8
"""Deep LSTM-based agent.

This implementation provides a simple LSTM agent that learns
from past price data to predict future price movement. It
uses only ``numpy`` for the network to avoid external
dependencies.
"""

from __future__ import annotations

import random
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

from ..logs.base import Logger
from ..market import Market
from ..order import LIMIT_ORDER
from ..order import Cancel
from ..order import Order
from .base import Agent


# シグモイド活性化関数
def sigmoid(x: np.ndarray) -> np.ndarray:
    """シグモイド関数を計算する．値を0から1の範囲に変換する．"""
    return 1 / (1 + np.exp(-x))


# シグモイド関数の導関数
def dsigmoid(y: np.ndarray) -> np.ndarray:
    """シグモイド関数の導関数を計算する．逆伝播で使用される．"""
    return y * (1 - y)


# ハイパボリックタンジェント活性化関数
def tanh(x: np.ndarray) -> np.ndarray:
    """ハイパボリックタンジェント関数を計算する．値を-1から1の範囲に変換する．"""
    return np.tanh(x)


# tanh関数の導関数
def dtanh(y: np.ndarray) -> np.ndarray:
    """tanh関数の導関数を計算する．逆伝播で使用される．"""
    return 1 - y**2


class SimpleLSTMNetwork:
    """シンプルなLSTMネットワーククラス．
    
    過去の価格データから未来の価格変動方向を予測するために使用される．
    LSTMセルと全結合層を組み合わせて2クラス分類を行う．
    """

    def __init__(self, input_size: int = 2, hidden_size: int = 32) -> None:
        """LSTMネットワークを初期化する．
        
        Args:
            input_size: 入力次元数（価格データとミッド価格の2次元）
            hidden_size: 隠れ層のユニット数
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        rng = np.random.default_rng(0)
        
        # LSTMゲートの重み行列を初期化する
        # Forget gate（忘却ゲート）: 過去の情報をどの程度忘れるかを制御
        self.Wf = rng.normal(scale=0.1, size=(hidden_size + input_size, hidden_size))
        self.bf = np.zeros(hidden_size)
        
        # Input gate（入力ゲート）: 新しい情報をどの程度取り入れるかを制御
        self.Wi = rng.normal(scale=0.1, size=(hidden_size + input_size, hidden_size))
        self.bi = np.zeros(hidden_size)
        
        # Output gate（出力ゲート）: セル状態からどの程度出力するかを制御
        self.Wo = rng.normal(scale=0.1, size=(hidden_size + input_size, hidden_size))
        self.bo = np.zeros(hidden_size)
        
        # Cell gate（セルゲート）: 新しい候補値を生成
        self.Wg = rng.normal(scale=0.1, size=(hidden_size + input_size, hidden_size))
        self.bg = np.zeros(hidden_size)
        
        # 全結合層の重み（分類用）
        self.W1 = rng.normal(scale=0.1, size=(hidden_size, hidden_size))
        self.b1 = np.zeros(hidden_size)
        self.W2 = rng.normal(scale=0.1, size=(hidden_size, 2))  # 2クラス分類
        self.b2 = np.zeros(2)
        
        # 学習率
        self.lr = 0.001

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Any]:
        """順伝播を実行する．

        Args:
            x: 入力系列データ，shape (seq_len, input_size)
        Returns:
            logits（分類スコア）とバックプロパゲーション用のキャッシュ
        """
        # 隠れ状態とセル状態を初期化
        h = np.zeros((x.shape[0] + 1, self.hidden_size))
        c = np.zeros((x.shape[0] + 1, self.hidden_size))
        caches = []
        
        # 各時刻でLSTMセルを実行
        for t in range(x.shape[0]):
            # 隠れ状態と入力を結合
            inp = np.concatenate([h[t], x[t]])
            
            # 各ゲートの出力を計算
            f = sigmoid(np.dot(inp, self.Wf) + self.bf)  # 忘却ゲート
            i = sigmoid(np.dot(inp, self.Wi) + self.bi)  # 入力ゲート
            o = sigmoid(np.dot(inp, self.Wo) + self.bo)  # 出力ゲート
            g = tanh(np.dot(inp, self.Wg) + self.bg)     # セルゲート
            
            # セル状態と隠れ状態を更新
            c[t + 1] = f * c[t] + i * g
            h[t + 1] = o * tanh(c[t + 1])
            
            # バックプロパゲーション用にキャッシュを保存
            caches.append((inp, f, i, o, g, c[t], c[t + 1]))
        
        # 最終的な隠れ状態を取得
        out = h[-1]
        
        # バッチ正規化を適用
        mu = out.mean()
        var = out.var() + 1e-6
        norm = (out - mu) / np.sqrt(var)
        
        # 全結合層を通して分類スコアを計算
        pre = np.dot(norm, self.W1) + self.b1
        relu = np.maximum(pre, 0)  # ReLU活性化
        logits = np.dot(relu, self.W2) + self.b2
        
        # キャッシュを作成
        cache = (caches, h, c, norm, mu, var, pre, relu)
        return logits, cache

    def backward(self, x: np.ndarray, y: int, cache: Any) -> None:
        """逆伝播を実行して重みを更新する．"""
        (caches, h, c, norm, mu, var, pre, relu) = cache
        
        # ソフトマックス損失の勾配を計算
        ex = np.exp(
            np.dot(relu, self.W2) + self.b2 - np.max(np.dot(relu, self.W2) + self.b2)
        )
        probs = ex / ex.sum()
        dlogits = probs
        dlogits[y] -= 1
        
        # 全結合層の勾配を計算
        dW2 = np.outer(relu, dlogits)
        db2 = dlogits
        drelu = np.dot(dlogits, self.W2.T)
        dpre = drelu * (pre > 0)  # ReLUの勾配
        dW1 = np.outer(norm, dpre)
        db1 = dpre
        
        # バッチ正規化の勾配を計算
        dnorm = np.dot(dpre, self.W1.T)
        dvar = (-0.5) * np.sum(dnorm * (h[-1] - mu)) / (var**1.5)
        dmu = -np.sum(dnorm) / np.sqrt(var) + dvar * (-2 * np.mean(h[-1] - mu))
        dh = (
            dnorm / np.sqrt(var)
            + dvar * 2 * (h[-1] - mu) / h[-1].size
            + dmu / h[-1].size
        )

        # LSTM層の勾配を逆向きに計算
        dh_next = dh
        dc_next = np.zeros(self.hidden_size)
        
        # 各重み行列の勾配を初期化
        dWf = np.zeros_like(self.Wf)
        dWi = np.zeros_like(self.Wi)
        dWo = np.zeros_like(self.Wo)
        dWg = np.zeros_like(self.Wg)
        dbf = np.zeros_like(self.bf)
        dbi = np.zeros_like(self.bi)
        dbo = np.zeros_like(self.bo)
        dbg = np.zeros_like(self.bg)
        
        # 時間を遡ってLSTMセルの勾配を計算
        for t in reversed(range(x.shape[0])):
            inp, f, i, o, g, c_prev, c_curr = caches[t]
            dh_cur = dh_next
            
            # 各ゲートの勾配を計算
            do = dh_cur * tanh(c_curr) * dsigmoid(o)
            dc = dh_cur * o * dtanh(tanh(c_curr)) + dc_next
            df = dc * c_prev * dsigmoid(f)
            di = dc * g * dsigmoid(i)
            dg = dc * i * dtanh(g)
            
            # 入力に対する勾配を計算
            dconcat = (
                np.dot(do, self.Wo.T)
                + np.dot(df, self.Wf.T)
                + np.dot(di, self.Wi.T)
                + np.dot(dg, self.Wg.T)
            )
            dh_next = dconcat[: self.hidden_size]
            dc_next = dc * f
            
            # 重み行列の勾配を累積
            dWf += np.outer(inp, df)
            dWi += np.outer(inp, di)
            dWo += np.outer(inp, do)
            dWg += np.outer(inp, dg)
            dbf += df
            dbi += di
            dbo += do
            dbg += dg
        
        # 勾配降下法で重みを更新
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
        """入力データに対する予測クラスを返す．
        
        Returns:
            0（下落予測）または1（上昇予測）
        """
        logits, _ = self.forward(x)
        return int(np.argmax(logits))

    def train(self, X: List[np.ndarray], y: List[int], epochs: int = 1) -> None:
        """モデルを訓練する．
        
        Args:
            X: 入力系列データのリスト
            y: ラベルのリスト
            epochs: エポック数
        """
        for _ in range(epochs):
            for x_s, y_s in zip(X, y):
                logits, cache = self.forward(x_s)
                self.backward(x_s, y_s, cache)

    def accuracy(self, X: List[np.ndarray], y: List[int]) -> float:
        """テストデータでの予測精度を計算する．
        
        Returns:
            正解率（0.0から1.0の間）
        """
        correct = 0
        for x_s, y_s in zip(X, y):
            pred = self.predict(x_s)
            if pred == y_s:
                correct += 1
        return correct / max(len(X), 1)


class DeepAgent(Agent):
    """LSTM予測モデルを内蔵したエージェントクラス．
    
    過去の価格データを学習して，未来の価格変動を予測し，
    それに基づいて売買注文を出すエージェント．
    """

    # 系列データの長さ（過去何時刻分のデータを使うか）
    sequence_length: int = 100
    # 予測する先の時刻（何時刻先の価格変動を予測するか）
    prediction_horizon: int = 100

    def __init__(
        self,
        agent_id: int,
        prng: random.Random,
        simulator: "Simulator",  # type: ignore  # NOQA
        name: str,
        logger: Optional[Logger] = None,
    ) -> None:
        """エージェントを初期化する．"""
        super().__init__(agent_id, prng, simulator, name, logger)
        # LSTMネットワークを作成
        self.net = SimpleLSTMNetwork()
        # 価格データの履歴を保存するリスト
        self.prices: List[float] = []
        self.mids: List[float] = []
        # ポジション履歴を保存するリスト
        self.position_history: List[int] = []

    def setup(
        self,
        settings: Dict[str, Any],
        accessible_markets_ids: List[int],
        *args: Any,
        **kwargs: Any,
    ) -> None:  # type: ignore
        """エージェントの設定を行う．現在は単一市場のみサポート．"""
        super().setup(settings=settings, accessible_markets_ids=accessible_markets_ids)
        if len(accessible_markets_ids) != 1:
            raise ValueError("DeepAgent currently supports one market")
        self.market_id = accessible_markets_ids[0]

    def _collect_data(self, market: Market) -> None:
        """市場から価格データを収集する．
        
        Args:
            market: 市場オブジェクト
        """
        self.prices.append(market.get_market_price())
        self.mids.append(market.get_mid_price() or market.get_market_price())

    def _prepare_dataset(self) -> Tuple[List[np.ndarray], List[int]]:
        """学習用データセットを準備する．
        
        過去の価格系列から特徴量を作成し，未来の価格変動方向をラベルとする．
        
        Returns:
            X: 特徴量リスト（価格とミッド価格の系列）
            y: ラベルリスト（0: 下落，1: 上昇）
        """
        X: List[np.ndarray] = []
        y: List[int] = []
        seq = self.sequence_length
        horizon = self.prediction_horizon
        
        # スライディングウィンドウで系列データを作成
        for t in range(len(self.mids) - seq - horizon):
            # 特徴量：過去seq時刻分の価格とミッド価格
            price_window = self.prices[t : t + seq]
            mid_window = self.mids[t : t + seq]
            arr = np.stack([price_window, mid_window], axis=1)
            
            # ラベル：horizon時刻先の価格変動方向
            future = self.mids[t + seq + horizon - 1]
            now = self.mids[t + seq - 1]
            label = 1 if future > now else 0  # 上昇なら1，下落なら0
            
            X.append(arr.astype(float))
            y.append(label)
        return X, y

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        """注文を提出する．LSTMの予測結果に基づいて売買判断を行う．
        
        Args:
            markets: 市場のリスト
            
        Returns:
            提出する注文のリスト
        """
        orders: List[Union[Order, Cancel]] = []
        market = markets[0]
        
        # 現在のポジションを記録
        self.position_history.append(self.asset_volumes.get(self.market_id, 0))
        
        # 価格データを収集
        self._collect_data(market)
        
        # 十分なデータが蓄積されるまで取引を行わない
        if len(self.mids) <= self.sequence_length + self.prediction_horizon + 100:
            return orders
        
        # 学習用データセットを準備
        X, y = self._prepare_dataset()
        if len(X) <= 100:
            return orders
        
        # 訓練用とテスト用にデータを分割
        train_X = X[:-100]
        train_y = y[:-100]
        test_X = X[-100:]
        test_y = y[-100:]
        
        # モデルを訓練
        if train_X:
            self.net.train(train_X, train_y, epochs=1)
        
        # テストデータで精度を評価
        acc = self.net.accuracy(test_X, test_y)
        
        # 精度が低い場合は取引を行わない
        if acc <= 0.51:
            return orders
        
        # 最新の価格データで予測を実行
        latest = np.stack(
            [self.prices[-self.sequence_length :], self.mids[-self.sequence_length :]],
            axis=1,
        ).astype(float)
        pred = self.net.predict(latest)
        
        # 現在のポジション
        volume = self.asset_volumes[self.market_id]
        
        # 予測に基づいて注文を生成
        if pred == 1 and volume < 1:  # 上昇予測かつロングポジションでない場合
            orders.append(
                Order(
                    agent_id=self.agent_id,
                    market_id=self.market_id,
                    is_buy=True,  # 買い注文
                    kind=LIMIT_ORDER,
                    volume=1,
                    price=market.get_market_price(),
                    ttl=1,
                )
            )
        if pred == 0 and volume > -1:  # 下落予測かつショートポジションでない場合
            orders.append(
                Order(
                    agent_id=self.agent_id,
                    market_id=self.market_id,
                    is_buy=False,  # 売り注文
                    kind=LIMIT_ORDER,
                    volume=1,
                    price=market.get_market_price(),
                    ttl=1,
                )
            )
        return orders

    def get_position_history(self) -> List[int]:
        """各時刻でのポジション履歴を返す．
        
        Returns:
            ポジション履歴のリスト
        """
        return self.position_history
