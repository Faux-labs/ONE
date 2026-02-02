
"""
Sui Trading Bot - Configuration
===============================

Network configurations and settings for the trading bot.
"""

from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class NetworkConfig:
    """Network configuration."""
    name: str
    graphql_url: str
    rpc_url: str
    ws_url: str
    faucet_url: Optional[str] = None
    chain_id: str = ""

    # Transaction settings
    default_gas_budget: int = 50_000_000
    max_gas_budget: int = 500_000_000
    gas_price_buffer: float = 1.1  # 10% buffer

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 2.0
    gas_multiplier: float = 1.2

    # Verification settings
    verification_timeout: float = 60.0
    confirmation_blocks: int = 1


# Network configurations
NETWORKS: Dict[str, NetworkConfig] = {
    "mainnet": NetworkConfig(
        name="mainnet",
        graphql_url="https://graphql.mainnet.sui.io/graphql",
        rpc_url="https://fullnode.mainnet.sui.io:443",
        ws_url="wss://fullnode.mainnet.sui.io:443",
        faucet_url=None,
        chain_id="35834a8a",
        default_gas_budget=50_000_000,
        max_gas_budget=500_000_000,
    ),
    "testnet": NetworkConfig(
        name="testnet",
        graphql_url="https://graphql.testnet.sui.io/graphql",
        rpc_url="https://fullnode.testnet.sui.io:443",
        ws_url="wss://fullnode.testnet.sui.io:443",
        faucet_url="https://faucet.testnet.sui.io/gas",
        chain_id="4c78adac",
        default_gas_budget=50_000_000,
        max_gas_budget=500_000_000,
    ),
    "devnet": NetworkConfig(
        name="devnet",
        graphql_url="https://graphql.devnet.sui.io/graphql",
        rpc_url="https://fullnode.devnet.sui.io:443",
        ws_url="wss://fullnode.devnet.sui.io:443",
        faucet_url="https://faucet.devnet.sui.io/gas",
        chain_id="0c8ec66d",
        default_gas_budget=50_000_000,
        max_gas_budget=500_000_000,
    ),
}


# DEX configurations (example DEX addresses)
DEX_CONFIGS: Dict[str, Dict[str, str]] = {
    "cetus": {
        "package": "0x...",  # Cetus DEX package address
        "module": "swap",
        "function_buy": "swap_x_to_y",
        "function_sell": "swap_y_to_x",
    },
    "turbos": {
        "package": "0x...",  # Turbos DEX package address
        "module": "router",
        "function_buy": "swap",
        "function_sell": "swap",
    },
    "kriya": {
        "package": "0x...",  # Kriya DEX package address
        "module": "swap",
        "function_buy": "swap_token_x",
        "function_sell": "swap_token_y",
    },
}


# Trading strategies configuration
STRATEGY_CONFIGS: Dict[str, Dict[str, Any]] = {
    "momentum": {
        "description": "Trade based on price momentum",
        "min_confidence": 0.75,
        "timeframes": ["5m", "15m", "1h"],
        "indicators": ["rsi", "macd", "volume"],
    },
    "mean_reversion": {
        "description": "Trade based on mean reversion",
        "min_confidence": 0.80,
        "timeframes": ["15m", "1h", "4h"],
        "indicators": ["bollinger_bands", "rsi", "stochastic"],
    },
    "breakout": {
        "description": "Trade based on price breakouts",
        "min_confidence": 0.85,
        "timeframes": ["1h", "4h", "1d"],
        "indicators": ["atr", "volume", "support_resistance"],
    },
    "grid_trading": {
        "description": "Grid-based trading strategy",
        "min_confidence": 0.60,
        "grid_levels": 10,
        "grid_spacing": 0.02,  # 2%
    },
}


# Risk management settings
RISK_CONFIG: Dict[str, Any] = {
    "max_position_size": 0.1,  # 10% of portfolio
    "max_daily_loss": 0.05,     # 5% of portfolio
    "max_trades_per_hour": 10,
    "stop_loss_percentage": 0.02,  # 2% stop loss
    "take_profit_percentage": 0.05,  # 5% take profit
    "slippage_tolerance": 0.01,  # 1% slippage
}


def get_network_config(network: str) -> NetworkConfig:
    """Get configuration for a network.

    Args:
        network: Network name

    Returns:
        Network configuration
    """
    if network not in NETWORKS:
        raise ValueError(f"Unknown network: {network}")
    return NETWORKS[network]


def get_dex_config(dex: str) -> Dict[str, str]:
    """Get configuration for a DEX.

    Args:
        dex: DEX name

    Returns:
        DEX configuration
    """
    if dex not in DEX_CONFIGS:
        raise ValueError(f"Unknown DEX: {dex}")
    return DEX_CONFIGS[dex]


def get_strategy_config(strategy: str) -> Dict[str, Any]:
    """Get configuration for a trading strategy.

    Args:
        strategy: Strategy name

    Returns:
        Strategy configuration
    """
    if strategy not in STRATEGY_CONFIGS:
        raise ValueError(f"Unknown strategy: {strategy}")
    return STRATEGY_CONFIGS[strategy]
