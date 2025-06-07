import random
from typing import Any, Dict, List, Optional, Tuple, Union

from .base import Agent
from ..market import Market
from ..order import LIMIT_ORDER, Cancel, Order


class PPOAgent(Agent):
    """A placeholder PPO-like agent.

    This implementation keeps policy and value estimates as dictionaries keyed by
    discretised market state. The update performed in :func:`submit_orders` is a
    very rough approximation of policy gradient with clipping, designed to avoid
    external dependencies while providing distinct behaviour from
    :class:`QLearningAgent` and :class:`DQNAgent`.
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
        self.policy: Dict[int, List[float]] = {}
        self.value: Dict[int, float] = {}
        self.epsilon: float = 0.1
        self.clip: float = 0.2

    def setup(
        self,
        settings: Dict[str, Any],
        accessible_markets_ids: List[int],
        *args: Any,
        **kwargs: Any,
    ) -> None:  # type: ignore
        super().setup(settings=settings, accessible_markets_ids=accessible_markets_ids)
        self.epsilon = settings.get("epsilon", 0.1)
        self.clip = settings.get("clip", 0.2)

    def _get_state(self, market: Market) -> int:
        diff = market.get_market_price() - market.get_fundamental_price()
        if diff > 0:
            return 1
        if diff < 0:
            return -1
        return 0

    def _choose_action(self, state: int) -> int:
        probs = self.policy.get(state, [1 / 3, 1 / 3, 1 / 3])
        r = self.prng.random()
        if r < probs[0]:
            return 0
        if r < probs[0] + probs[1]:
            return 1
        return 2

    def _update(self, state: int, action: int, reward: float) -> None:
        probs = self.policy.get(state, [1 / 3, 1 / 3, 1 / 3])
        value = self.value.get(state, 0.0)
        advantage = reward - value
        new_probs = list(probs)
        new_probs[action] += self.epsilon * advantage
        total = sum(max(p, 1e-6) for p in new_probs)
        new_probs = [max(p, 1e-6) / total for p in new_probs]
        ratio = new_probs[action] / probs[action]
        clipped_ratio = max(min(ratio, 1 + self.clip), 1 - self.clip)
        self.policy[state] = [
            (1 - clipped_ratio) * p + clipped_ratio * np  # type: ignore
            for p, np in zip(probs, new_probs)
        ]
        self.value[state] = value + 0.1 * advantage

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        orders: List[Union[Order, Cancel]] = []
        if not markets:
            return orders
        market = markets[0]
        state = self._get_state(market)
        action = self._choose_action(state)
        reward = -abs(market.get_market_price() - market.get_fundamental_price())
        self._update(state, action, reward)

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
