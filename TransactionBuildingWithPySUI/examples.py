
"""
Sui Trading Bot - Examples and Usage Guide
===========================================

This script demonstrates various features of the Sui Trading Bot implementation.
"""

import asyncio
import os
from sui_trading_bot import (
    Wallet, TradingBot, TradingSignal, TradingDecision,
    GasConfig, TransactionResult, SuiGraphQLClient
)

# ============================================================================
# Example 1: Wallet Creation and Management
# ============================================================================

def example_wallet_management():
    """Demonstrate wallet creation and management."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Wallet Creation and Management")
    print("="*60)

    # Create a new wallet
    wallet = Wallet.create_new()
    print(f"\n✓ Created new wallet")
    print(f"  Address: {wallet.address}")
    print(f"  Public Key: {wallet.public_key[:20]}..." if wallet.public_key else "  Public Key: None")

    # Save wallet info (in production, encrypt this!)
    wallet_info = {
        "address": wallet.address,
        "private_key": wallet.private_key,
        "public_key": wallet.public_key
    }
    print(f"\n  Wallet Info: {wallet_info}")

    # Load wallet from private key
    if wallet.private_key:
        loaded_wallet = Wallet.from_private_key(wallet.private_key)
        print(f"\n✓ Loaded wallet from private key")
        print(f"  Address matches: {loaded_wallet.address == wallet.address}")

    return wallet


# ============================================================================
# Example 2: GraphQL Client Usage
# ============================================================================

async def example_graphql_client():
    """Demonstrate GraphQL client usage."""
    print("\n" + "="*60)
    print("EXAMPLE 2: GraphQL Client Usage")
    print("="*60)

    # Create client for testnet
    client = SuiGraphQLClient(network="testnet")
    print(f"\n✓ Created GraphQL client")
    print(f"  Endpoint: {client.endpoint}")
    print(f"  Network: {client.network}")

    # Example query (would need valid address for real query)
    test_address = "0x1234567890abcdef1234567890abcdef12345678"

    try:
        balance = client.get_balance_sync(test_address)
        print(f"\n✓ Queried balance for {test_address[:20]}...")
        print(f"  Balance: {balance} MIST")
    except Exception as e:
        print(f"\n  Note: Query failed as expected with test address: {e}")

    return client


# ============================================================================
# Example 3: Balance Checking
# ============================================================================

async def example_balance_checking(wallet: Wallet):
    """Demonstrate balance checking."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Balance Checking")
    print("="*60)

    bot = TradingBot(wallet, network="testnet")

    # Check SUI balance
    print(f"\n  Checking balance for {wallet.address[:30]}...")

    try:
        balance = await bot.check_balance()
        print(f"✓ Balance: {balance / 1_000_000_000:.9f} SUI")
    except Exception as e:
        print(f"  Note: Balance check failed (expected for new wallet): {e}")

    return bot


# ============================================================================
# Example 4: Airdrop Request
# ============================================================================

async def example_airdrop(wallet: Wallet):
    """Demonstrate airdrop request."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Airdrop Request")
    print("="*60)

    bot = TradingBot(wallet, network="testnet")

    print(f"\n  Requesting airdrop for {wallet.address[:30]}...")
    print(f"  Amount: 1 SUI (1,000,000,000 MIST)")

    try:
        success = await bot.request_airdrop()
        if success:
            print("✓ Airdrop request successful!")
            print("  Tokens should arrive shortly.")
        else:
            print("✗ Airdrop request failed")
    except Exception as e:
        print(f"  Note: Airdrop failed: {e}")

    return bot


# ============================================================================
# Example 5: Trading Signal Creation
# ============================================================================

def example_trading_signals():
    """Demonstrate trading signal creation."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Trading Signal Creation")
    print("="*60)

    # BUY signal
    buy_signal = TradingSignal(
        decision=TradingDecision.BUY,
        amount=0.5,  # 0.5 SUI
        confidence=0.92,
        token_address="0x3dcfc5338d8358450b145629c985a9d6cb20f9c0ab6667e328e152cdfd8022cd",
        metadata={
            "strategy": "momentum",
            "indicators": ["rsi", "macd"],
            "timeframe": "1h"
        }
    )

    print(f"\n✓ Created BUY signal")
    print(f"  Decision: {buy_signal.decision.value}")
    print(f"  Amount: {buy_signal.amount} SUI")
    print(f"  Confidence: {buy_signal.confidence * 100}%")
    print(f"  Token: {buy_signal.token_address[:40]}...")
    print(f"  Metadata: {buy_signal.metadata}")

    # SELL signal
    sell_signal = TradingSignal(
        decision=TradingDecision.SELL,
        amount=100.0,  # 100 tokens
        confidence=0.85,
        token_address="0x3dcfc5338d8358450b145629c985a9d6cb20f9c0ab6667e328e152cdfd8022cd",
        metadata={
            "token_object_id": "0xabc123...",  # Required for SELL
            "strategy": "take_profit",
            "entry_price": 0.01
        }
    )

    print(f"\n✓ Created SELL signal")
    print(f"  Decision: {sell_signal.decision.value}")
    print(f"  Amount: {sell_signal.amount} tokens")
    print(f"  Confidence: {sell_signal.confidence * 100}%")

    # HOLD signal
    hold_signal = TradingSignal(
        decision=TradingDecision.HOLD,
        amount=0.0,
        confidence=0.45,
        token_address="0x3dcfc5338d8358450b145629c985a9d6cb20f9c0ab6667e328e152cdfd8022cd",
        metadata={"reason": "insufficient_confidence"}
    )

    print(f"\n✓ Created HOLD signal")
    print(f"  Decision: {hold_signal.decision.value}")
    print(f"  Confidence: {hold_signal.confidence * 100}%")

    return buy_signal, sell_signal, hold_signal


# ============================================================================
# Example 6: Gas Configuration
# ============================================================================

def example_gas_configuration():
    """Demonstrate gas configuration."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Gas Configuration")
    print("="*60)

    # Default gas config
    default_gas = GasConfig()
    print(f"\n✓ Default Gas Config")
    print(f"  Budget: {default_gas.budget} MIST ({default_gas.budget / 1_000_000_000} SUI)")
    print(f"  Price: {default_gas.price}")

    # Custom gas config
    custom_gas = GasConfig(
        budget=100_000_000,  # 0.1 SUI
        price=2000  # Higher gas price for faster confirmation
    )
    print(f"\n✓ Custom Gas Config")
    print(f"  Budget: {custom_gas.budget} MIST ({custom_gas.budget / 1_000_000_000} SUI)")
    print(f"  Price: {custom_gas.price}")

    # Gas config for high priority
    priority_gas = GasConfig(
        budget=200_000_000,
        price=5000
    )
    print(f"\n✓ High Priority Gas Config")
    print(f"  Budget: {priority_gas.budget} MIST")
    print(f"  Price: {priority_gas.price}")

    return default_gas, custom_gas, priority_gas


# ============================================================================
# Example 7: Transaction Building (Mock)
# ============================================================================

async def example_transaction_building(wallet: Wallet):
    """Demonstrate transaction building."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Transaction Building")
    print("="*60)

    client = SuiGraphQLClient(network="testnet")
    builder = bot.tx_builder if 'bot' in globals() else None

    if not builder:
        from sui_trading_bot import TransactionBuilder
        builder = TransactionBuilder(client, wallet)

    # Build BUY transaction
    print(f"\n  Building BUY transaction...")
    try:
        buy_tx = await builder.build_buy_transaction(
            token_address="0x1234...",
            amount=0.1,
            slippage=0.01,
            gas_config=GasConfig()
        )
        print(f"✓ BUY transaction built")
        print(f"  Sender: {buy_tx['sender'][:40]}...")
        print(f"  Commands: {len(buy_tx['commands'])}")
        print(f"  Gas Budget: {buy_tx['gas']['budget']}")
    except Exception as e:
        print(f"  Note: {e}")

    # Build SELL transaction
    print(f"\n  Building SELL transaction...")
    try:
        sell_tx = await builder.build_sell_transaction(
            token_address="0x1234...",
            token_object_id="0xabcd...",
            amount=50.0,
            slippage=0.02,
            gas_config=GasConfig()
        )
        print(f"✓ SELL transaction built")
        print(f"  Commands: {len(sell_tx['commands'])}")
    except Exception as e:
        print(f"  Note: {e}")

    return builder


# ============================================================================
# Example 8: Complete Trading Flow (Mock)
# ============================================================================

async def example_complete_trading_flow():
    """Demonstrate complete trading flow."""
    print("\n" + "="*60)
    print("EXAMPLE 8: Complete Trading Flow (Simulation)")
    print("="*60)

    # Create wallet
    wallet = Wallet.create_new()
    print(f"\n✓ Wallet created: {wallet.address[:40]}...")

    # Initialize bot
    bot = TradingBot(wallet, network="testnet")
    print(f"✓ Trading bot initialized")

    # Create trading signal
    signal = TradingSignal(
        decision=TradingDecision.BUY,
        amount=0.5,
        confidence=0.88,
        token_address="0x3dcfc5338d8358450b145629c985a9d6cb20f9c0ab6667e328e152cdfd8022cd",
        metadata={"strategy": "breakout", "timeframe": "15m"}
    )
    print(f"\n✓ Trading signal created")
    print(f"  Type: {signal.decision.value}")
    print(f"  Amount: {signal.amount} SUI")
    print(f"  Confidence: {signal.confidence * 100}%")

    # Get optimal gas config
    print(f"\n  Getting optimal gas configuration...")
    try:
        gas_config = await bot.get_optimal_gas_config()
        print(f"✓ Gas config: price={gas_config.price}, budget={gas_config.budget}")
    except Exception as e:
        print(f"  Using default gas config: {e}")
        gas_config = GasConfig()

    # Simulate trade execution
    print(f"\n  Simulating trade execution...")
    print(f"  This would execute the transaction on-chain with retry logic")
    print(f"  and verify the result.")

    # Mock result
    result = TransactionResult(
        success=True,
        digest="0x" + "a" * 64,
        gas_used=2_500_000,
        retry_count=0
    )

    print(f"\n✓ Trade simulated")
    print(f"  Success: {result.success}")
    print(f"  Digest: {result.digest[:50]}...")
    print(f"  Gas Used: {result.gas_used} MIST")
    print(f"  Retries: {result.retry_count}")

    return result


# ============================================================================
# Example 9: Error Handling and Retry Logic
# ============================================================================

async def example_error_handling():
    """Demonstrate error handling and retry logic."""
    print("\n" + "="*60)
    print("EXAMPLE 9: Error Handling and Retry Logic")
    print("="*60)

    print(f"\n  The TransactionExecutor handles various failure scenarios:")
    print(f"  1. Network timeouts - retries with exponential backoff")
    print(f"  2. Insufficient gas - increases gas budget on retry")
    print(f"  3. Transaction failures - verifies on-chain status")
    print(f"  4. Verification timeouts - configurable timeout periods")

    print(f"\n  Retry Configuration:")
    print(f"  - Max Retries: 3")
    print(f"  - Retry Delay: 2.0 seconds (increases with each retry)")
    print(f"  - Gas Multiplier: 1.2x on each retry")

    # Simulate failed then successful transaction
    print(f"\n  Simulating: Transaction fails twice, succeeds on third attempt")

    for attempt in range(1, 4):
        print(f"\n  Attempt {attempt}:")
        if attempt < 3:
            print(f"    ✗ Failed (simulated)")
            print(f"    → Waiting {attempt * 2} seconds before retry...")
            print(f"    → Increasing gas budget by 20%")
        else:
            print(f"    ✓ Success!")
            print(f"    → Transaction verified on-chain")

    print(f"\n  Final gas budget: 50M * 1.2^2 = 72M MIST")


# ============================================================================
# Example 10: Production Setup
# ============================================================================

def example_production_setup():
    """Show production setup recommendations."""
    print("\n" + "="*60)
    print("EXAMPLE 10: Production Setup")
    print("="*60)

    print(f"\n  1. Environment Variables:")
    print(f"     export SUI_PRIVATE_KEY='your_encrypted_private_key'")
    print(f"     export SUI_NETWORK='mainnet'")
    print(f"     export SUI_RPC_URL='https://your-rpc-endpoint.com'")

    print(f"\n  2. Secure Key Management:")
    print(f"     - Use hardware security modules (HSM)")
    print(f"     - Encrypt private keys at rest")
    print(f"     - Use environment variables or secret managers")
    print(f"     - Never commit keys to version control")

    print(f"\n  3. Monitoring and Alerting:")
    print(f"     - Track transaction success rates")
    print(f"     - Monitor gas costs")
    print(f"     - Set up alerts for failed transactions")
    print(f"     - Log all trading decisions")

    print(f"\n  4. Risk Management:")
    print(f"     - Set maximum trade sizes")
    print(f"     - Implement circuit breakers")
    print(f"     - Use stop-loss mechanisms")
    print(f"     - Monitor portfolio exposure")

    print(f"\n  5. Code Example:")
    print(f"     ```python")
    print(f"     import os")
    print(f"     from sui_trading_bot import TradingBot, Wallet")
    print(f"     ")
    print(f"     # Load from environment")
    print(f"     private_key = os.environ['SUI_PRIVATE_KEY']")
    print(f"     network = os.environ.get('SUI_NETWORK', 'mainnet')")
    print(f"     ")
    print(f"     # Create wallet and bot")
    print(f"     wallet = Wallet.from_private_key(private_key)")
    print(f"     bot = TradingBot(wallet, network=network)")
    print(f"     ")
    print(f"     # Execute trades...")
    print(f"     ```")


# ============================================================================
# Main runner
# ============================================================================

async def run_all_examples():
    """Run all examples."""
    print("\n" + "="*60)
    print("SUI TRADING BOT - COMPREHENSIVE EXAMPLES")
    print("="*60)

    # Run synchronous examples
    wallet = example_wallet_management()
    buy_sig, sell_sig, hold_sig = example_trading_signals()
    default_gas, custom_gas, priority_gas = example_gas_configuration()
    example_production_setup()

    # Run async examples
    await example_graphql_client()
    await example_balance_checking(wallet)
    await example_airdrop(wallet)
    await example_transaction_building(wallet)
    await example_complete_trading_flow()
    await example_error_handling()

    print("\n" + "="*60)
    print("ALL EXAMPLES COMPLETED")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(run_all_examples())
