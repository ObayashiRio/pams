#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
複数エージェント混在シミュレーション実行スクリプト

FCN 30%、Test 30%、DeepAgent 10%、MarketMaker 30%の割合で
エージェントを配置してシミュレーションを実行する。
"""

import json
import random
from typing import List

from pams.logs.market_step_loggers import MarketStepPrintLogger
from pams.runners.sequential import SequentialRunner


class CustomLogger(MarketStepPrintLogger):
    """カスタムロガー：価格履歴を記録する"""
    def __init__(self):
        super().__init__()
        self.prices: List[float] = []
        self.volumes: List[int] = []

    def process_market_step_end_log(self, log) -> None:
        super().process_market_step_end_log(log)
        self.prices.append(log.market.get_market_price())
        self.volumes.append(log.market.get_executed_volume())


def main():
    print("=== 複数エージェント混在シミュレーション開始 ===")
    
    # カスタムロガーを作成
    logger = CustomLogger()
    
    # SequentialRunnerを作成
    runner = SequentialRunner(
        settings="mixed_agents_config.json",
        prng=random.Random(42),  # 再現性のためseed固定
        logger=logger,
    )
    
    # 設定ファイルから情報を読み込んで表示
    with open("mixed_agents_config.json", "r") as f:
        config = json.load(f)
    
    print(f"FCNAgent: {config['FCNAgents']['numAgents']}体 (30%)")
    print(f"TestAgent: {config['TestAgents']['numAgents']}体 (30%) - ランダム取引")
    print(f"DeepAgent: {config['DeepAgents']['numAgents']}体 (10%)")
    print(f"MarketMakerAgent: {config['MarketMakerAgents']['numAgents']}体 (30%)")
    print("=" * 50)
    
    # シミュレーション実行
    runner.main()
    
    print("\n=== シミュレーション完了 ===")
    
    # 結果の分析
    initial_price = config['Market']['marketPrice']
    final_price = logger.prices[-1] if logger.prices else initial_price
    total_volume = sum(logger.volumes)
    
    print(f"初期価格: {initial_price:.2f}")
    print(f"最終価格: {final_price:.2f}")
    price_change = ((final_price - initial_price) / initial_price) * 100
    print(f"価格変動率: {price_change:.2f}%")
    print(f"総取引量: {total_volume}")
    print(f"取引ステップ数: {len(logger.prices)}")


if __name__ == "__main__":
    main() 