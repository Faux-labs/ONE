
"""
Sui Trading Bot
===============

A comprehensive Python implementation for building and executing trading 
transactions on the Sui blockchain using GraphQL and pysui.

Features:
- Wallet creation and management
- GraphQL client for Sui blockchain queries  
- BUY/SELL transaction building
- Gas estimation and fee optimization
- Retry logic for failed transactions
- On-chain transaction verification
- Airdrop and balance checking

Quick Start:
    >>> from sui_trading_bot import Wallet, TradingBot, TradingSignal, TradingDecision
    >>> 
    >>> # Create wallet
    >>> wallet = Wallet.create_new()
    >>> 
    >>> # Initialize bot
    >>> bot = TradingBot(wallet, network="testnet")
    >>> 
    >>> # Check balance
    >>> balance = await bot.check_balance()
    >>> 
    >>> # Execute trade
    >>> signal = TradingSignal(
    ...     decision=TradingDecision.BUY,
    ...     amount=0.5,
    ...     confidence=0.85,
    ...     token_address="0x..."
    ... )
    >>> result = await bot.execute_trade(signal)
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"

# Core classes
from sui_trading_bot.sui_trading_bot import (
    Wallet,
    TradingBot,
    TradingSignal,
    TradingDecision,
    GasConfig,
    TransactionResult,
    SuiGraphQLClient,
    TransactionBuilder,
    TransactionExecutor,
    create_wallet,
    load_wallet,
    create_trading_bot,
)

# Configuration
from sui_trading_bot.config import (
    NetworkConfig,
    NETWORKS,
    DEX_CONFIGS,
    STRATEGY_CONFIGS,
    RISK_CONFIG,
    get_network_config,
    get_dex_config,
    get_strategy_config,
)

# Optional pysui integration
try:
    from sui_trading_bot.pysui_integration import (
        PYSUI_AVAILABLE,
        PySUIWallet,
        PySUITradingBot,
        create_pysui_bot,
    )
except ImportError:
    PYSUI_AVAILABLE = False
    PySUIWallet = None
    PySUITradingBot = None
    create_pysui_bot = None

__all__ = [
    # Version
    "__version__",

    # Core classes
    "Wallet",
    "TradingBot", 
    "TradingSignal",
    "TradingDecision",
    "GasConfig",
    "TransactionResult",
    "SuiGraphQLClient",
    "TransactionBuilder",
    "TransactionExecutor",

    # Factory functions
    "create_wallet",
    "load_wallet",
    "create_trading_bot",

    # Configuration
    "NetworkConfig",
    "NETWORKS",
    "DEX_CONFIGS",
    "STRATEGY_CONFIGS",
    "RISK_CONFIG",
    "get_network_config",
    "get_dex_config",
    "get_strategy_config",

    # PySUI integration (optional)
    "PYSUI_AVAILABLE",
    "PySUIWallet",
    "PySUITradingBot",
    "create_pysui_bot",
]
