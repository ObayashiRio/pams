import argparse
import json
import random
from typing import List
from typing import Optional

import matplotlib.pyplot as plt

from pams.agents.deep_agent import DeepAgent
from pams.logs.market_step_loggers import MarketStepPrintLogger
from pams.runners.sequential import SequentialRunner


class PriceLogger(MarketStepPrintLogger):
    def __init__(self):
        super().__init__()
        self.prices: List[float] = []

    def process_market_step_end_log(self, log) -> None:
        super().process_market_step_end_log(log)
        self.prices.append(log.market.get_market_price())


def plot_simulation_results(runner: SequentialRunner, prices: List[float]) -> None:
    """Plot simulation results including market prices and agent positions."""
    steps = range(len(prices))

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot price history
    ax1.plot(steps, prices, label="Market Price")
    ax1.set_title("Market Price History")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Price")
    ax1.grid(True)
    ax1.legend()

    # Plot agent positions
    for agent in runner.simulator.agents:
        if isinstance(agent, DeepAgent):
            positions = agent.get_position_history()
            ax2.plot(steps, positions, label=f"Agent {agent.agent_id}")

    ax2.set_title("Agent Positions")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Position")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, required=True, help="config.json file"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=None, help="simulation random seed"
    )
    args = parser.parse_args()
    config: str = args.config
    seed: Optional[int] = args.seed

    price_logger = PriceLogger()
    runner = SequentialRunner(
        settings=config,
        prng=random.Random(seed) if seed is not None else None,
        logger=price_logger,
    )
    runner.main()

    # Plot simulation results
    plot_simulation_results(runner, price_logger.prices)


if __name__ == "__main__":
    main()
