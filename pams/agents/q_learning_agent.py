import random
from typing import Any, Dict, List, Optional, Tuple, Union

from .base import Agent
from ..market import Market
from ..order import LIMIT_ORDER, Cancel, Order


class QLearningAgent(Agent):
    """Simple Q-Learning agent.

    This implementation uses a small discrete state space based on the sign of
    the gap between market price and fundamental price. Actions are buy, sell or
    do nothing. Q-values are kept in a dictionary and updated in a very naive
    way whenever :func:`submit_orders` is called.
    """

    def __init__(
        self,
        agent_id: int,
        prng: random.Random,
        simulator: "Simulator",  # type: ignore  # NOQA
        name: str,
        logger: Optional["Logger"] = None,  # type: ignore  # NOQA
    ) -> None:
        super().__init__(agent_id=agent_id, prng=prng, simulator=simulator, name=name, logger=logger)
        self.q_table: Dict[Tuple[int, int], float] = {}
        self.epsilon: float = 0.1
        self.alpha: float = 0.1
        self.gamma: float = 0.95
        self.prev_state: Optional[int] = None
        self.prev_action: Optional[int] = None

    def setup(
        self,
        settings: Dict[str, Any],
        accessible_markets_ids: List[int],
        *args: Any,
        **kwargs: Any,
    ) -> None:  # type: ignore
        super().setup(settings=settings, accessible_markets_ids=accessible_markets_ids)
        self.epsilon = settings.get("epsilon", 0.1)
        self.alpha = settings.get("alpha", 0.1)
        self.gamma = settings.get("gamma", 0.95)

    def _get_state(self, market: Market) -> int:
        diff = market.get_market_price() - market.get_fundamental_price()
        if diff > 0:
            return 1
        if diff < 0:
            return -1
        return 0

    def _choose_action(self, state: int) -> int:
        if self.prng.random() < self.epsilon:
            return self.prng.choice([0, 1, 2])  # 0: buy, 1: sell, 2: hold
        qs = [self.q_table.get((state, a), 0.0) for a in [0, 1, 2]]
        max_q = max(qs)
        actions = [a for a, q in enumerate(qs) if q == max_q]
        return self.prng.choice(actions)

    def _update_q(self, reward: float, state: int) -> None:
        if self.prev_state is None or self.prev_action is None:
            return
        prev_key = (self.prev_state, self.prev_action)
        prev_q = self.q_table.get(prev_key, 0.0)
        future_qs = [self.q_table.get((state, a), 0.0) for a in [0, 1, 2]]
        self.q_table[prev_key] = prev_q + self.alpha * (
            reward + self.gamma * max(future_qs) - prev_q
        )

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        orders: List[Union[Order, Cancel]] = []
        if not markets:
            return orders
        market = markets[0]
        state = self._get_state(market)
        action = self._choose_action(state)
        reward = -abs(market.get_market_price() - market.get_fundamental_price())
        self._update_q(reward, state)
        self.prev_state = state
        self.prev_action = action

        price = market.get_market_price()
        if action == 0:
            orders.append(
                Order(
                    agent_id=self.agent_id,
                    market_id=market.market_id,
                    is_buy=True,
                    kind=LIMIT_ORDER,
                    volume=1,
                    price=price,
                    ttl=1,
                )
            )
        elif action == 1:
            orders.append(
                Order(
                    agent_id=self.agent_id,
                    market_id=market.market_id,
                    is_buy=False,
                    kind=LIMIT_ORDER,
                    volume=1,
                    price=price,
                    ttl=1,
                )
            )
        return orders
