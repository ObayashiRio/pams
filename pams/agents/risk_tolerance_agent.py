import random
from typing import Any, Dict, List, Optional, Union

from .base import Agent
from ..market import Market
from ..order import LIMIT_ORDER, Cancel, Order


class RiskToleranceAgent(Agent):
    """Agent that changes its behavior depending on risk tolerance."""

    def __init__(
        self,
        agent_id: int,
        prng: random.Random,
        simulator: "Simulator",  # type: ignore  # NOQA
        name: str,
        logger: Optional["Logger"] = None,  # type: ignore  # NOQA
    ) -> None:
        super().__init__(agent_id=agent_id, prng=prng, simulator=simulator, name=name, logger=logger)
        self.risk_type: str = "neutral"

    def setup(
        self,
        settings: Dict[str, Any],
        accessible_markets_ids: List[int],
        *args: Any,
        **kwargs: Any,
    ) -> None:  # type: ignore
        super().setup(settings=settings, accessible_markets_ids=accessible_markets_ids)
        self.risk_type = settings.get("riskType", "neutral")

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        orders: List[Union[Order, Cancel]] = []
        margin_scale: float = 10.0
        volume_scale: int = 100
        time_length_scale: int = 100
        buy_chance: float = 0.4
        sell_chance: float = 0.4

        if self.risk_type == "averse":
            margin_scale *= 0.5
            volume_scale = int(volume_scale * 0.5)
        elif self.risk_type == "loving":
            margin_scale *= 1.5
            volume_scale = int(volume_scale * 1.5)

        for market in markets:
            if self.is_market_accessible(market_id=market.market_id):
                price = market.get_market_price() + (
                    self.prng.random() * 2 * margin_scale - margin_scale
                )
                volume = self.prng.randint(1, volume_scale)
                time_length = self.prng.randint(1, time_length_scale)
                p = self.prng.random()
                if p < buy_chance + sell_chance:
                    orders.append(
                        Order(
                            agent_id=self.agent_id,
                            market_id=market.market_id,
                            is_buy=p < buy_chance,
                            kind=LIMIT_ORDER,
                            volume=volume,
                            price=price,
                            ttl=time_length,
                        )
                    )
        return orders

