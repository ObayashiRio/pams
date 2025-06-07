import random
from typing import List, cast

import pytest

from pams import LIMIT_ORDER, Market, Order, Simulator
from pams.agents import RiskToleranceAgent
from pams.logs import Logger


class TestRiskToleranceAgent:
    @pytest.mark.parametrize("risk_type", ["averse", "neutral", "loving"])
    def test_submit_orders(self, risk_type: str) -> None:
        seed = 42
        sim = Simulator(prng=random.Random(seed + 2))
        logger = Logger()
        agent = RiskToleranceAgent(
            agent_id=1,
            prng=random.Random(seed),
            simulator=sim,
            name="risk_agent",
            logger=logger,
        )
        settings = {"assetVolume": 50, "cashAmount": 10000, "riskType": risk_type}
        agent.setup(settings=settings, accessible_markets_ids=[0])
        market = Market(
            market_id=0,
            prng=random.Random(seed + 1),
            simulator=sim,
            name="market",
            logger=logger,
        )
        market._update_time(next_fundamental_price=300.0)

        orders = cast(List[Order], agent.submit_orders(markets=[market]))
        _prng = random.Random(seed)
        margin_scale = 10.0
        volume_scale = 100
        if risk_type == "averse":
            margin_scale *= 0.5
            volume_scale = int(volume_scale * 0.5)
        elif risk_type == "loving":
            margin_scale *= 1.5
            volume_scale = int(volume_scale * 1.5)
        price = 300 + _prng.random() * 2 * margin_scale - margin_scale
        volume = _prng.randint(1, volume_scale)
        time_length = _prng.randint(1, 100)
        p = _prng.random()
        if p >= 0.8:
            assert len(orders) == 0
        else:
            assert len(orders) == 1
            assert orders[0].agent_id == 1
            assert orders[0].market_id == 0
            assert orders[0].is_buy == (p < 0.4)
            assert orders[0].kind == LIMIT_ORDER
            assert orders[0].volume == volume
            assert orders[0].price == price
            assert orders[0].ttl == time_length

