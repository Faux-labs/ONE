
"""
Sui Trading Bot - Test Suite
============================

Unit tests for the trading bot implementation.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass

from sui_trading_bot import (
    Wallet, TradingBot, TradingSignal, TradingDecision,
    GasConfig, TransactionResult, SuiGraphQLClient,
    TransactionBuilder, TransactionExecutor
)


# ============================================================================
# Wallet Tests
# ============================================================================

class TestWallet:
    """Test wallet functionality."""

    def test_create_wallet(self):
        """Test wallet creation."""
        wallet = Wallet.create_new()
        assert wallet.address.startswith("0x")
        assert len(wallet.address) == 42  # 0x + 40 hex chars
        assert wallet.private_key is not None
        assert wallet.public_key is not None

    def test_wallet_from_private_key(self):
        """Test loading wallet from private key."""
        original = Wallet.create_new()
        loaded = Wallet.from_private_key(original.private_key)
        assert loaded.address == original.address

    def test_sign_transaction(self):
        """Test transaction signing."""
        wallet = Wallet.create_new()
        tx_bytes = b"test transaction"
        signature = wallet.sign_transaction(tx_bytes)
        assert signature is not None
        assert len(signature) > 0


# ============================================================================
# Trading Signal Tests
# ============================================================================

class TestTradingSignal:
    """Test trading signals."""

    def test_buy_signal(self):
        """Test BUY signal creation."""
        signal = TradingSignal(
            decision=TradingDecision.BUY,
            amount=0.5,
            confidence=0.85,
            token_address="0x1234...",
            metadata={"strategy": "test"}
        )
        assert signal.decision == TradingDecision.BUY
        assert signal.amount == 0.5
        assert signal.confidence == 0.85

    def test_sell_signal(self):
        """Test SELL signal creation."""
        signal = TradingSignal(
            decision=TradingDecision.SELL,
            amount=100.0,
            confidence=0.90,
            token_address="0x1234...",
            metadata={"token_object_id": "0xabcd..."}
        )
        assert signal.decision == TradingDecision.SELL
        assert signal.amount == 100.0

    def test_hold_signal(self):
        """Test HOLD signal creation."""
        signal = TradingSignal(
            decision=TradingDecision.HOLD,
            amount=0.0,
            confidence=0.30,
            token_address="0x1234...",
            metadata={}
        )
        assert signal.decision == TradingDecision.HOLD


# ============================================================================
# Gas Configuration Tests
# ============================================================================

class TestGasConfig:
    """Test gas configuration."""

    def test_default_gas_config(self):
        """Test default gas configuration."""
        gas = GasConfig()
        assert gas.budget == 50_000_000
        assert gas.price is None
        assert gas.payment is None

    def test_custom_gas_config(self):
        """Test custom gas configuration."""
        gas = GasConfig(
            budget=100_000_000,
            price=2000,
            payment=[{"objectId": "0x1234"}]
        )
        assert gas.budget == 100_000_000
        assert gas.price == 2000
        assert gas.payment is not None

    def test_gas_to_dict(self):
        """Test gas config serialization."""
        gas = GasConfig(budget=75_000_000, price=1500)
        d = gas.to_dict()
        assert d["budget"] == 75_000_000
        assert d["price"] == 1500


# ============================================================================
# Transaction Result Tests
# ============================================================================

class TestTransactionResult:
    """Test transaction results."""

    def test_successful_result(self):
        """Test successful transaction result."""
        result = TransactionResult(
            success=True,
            digest="0x" + "a" * 64,
            gas_used=2_500_000,
            retry_count=0
        )
        assert result.success is True
        assert result.digest is not None
        assert result.gas_used == 2_500_000

    def test_failed_result(self):
        """Test failed transaction result."""
        result = TransactionResult(
            success=False,
            errors=["Insufficient funds"],
            retry_count=3
        )
        assert result.success is False
        assert len(result.errors) > 0
        assert result.retry_count == 3

    def test_result_to_dict(self):
        """Test result serialization."""
        result = TransactionResult(
            success=True,
            digest="0xabc...",
            gas_used=1_000_000
        )
        d = result.to_dict()
        assert d["success"] is True
        assert "digest" in d
        assert "timestamp" in d


# ============================================================================
# GraphQL Client Tests
# ============================================================================

class TestSuiGraphQLClient:
    """Test GraphQL client."""

    def test_client_initialization(self):
        """Test client initialization."""
        client = SuiGraphQLClient(network="testnet")
        assert client.network == "testnet"
        assert "testnet" in client.endpoint

    def test_mainnet_endpoint(self):
        """Test mainnet endpoint."""
        client = SuiGraphQLClient(network="mainnet")
        assert "mainnet" in client.endpoint

    def test_custom_endpoint(self):
        """Test custom endpoint."""
        custom = "https://custom.sui.io/graphql"
        client = SuiGraphQLClient(endpoint=custom)
        assert client.endpoint == custom


# ============================================================================
# Transaction Builder Tests
# ============================================================================

class TestTransactionBuilder:
    """Test transaction builder."""

    @pytest.fixture
    def builder(self):
        """Create test builder."""
        client = SuiGraphQLClient(network="testnet")
        wallet = Wallet.create_new()
        return TransactionBuilder(client, wallet)

    @pytest.mark.asyncio
    async def test_build_buy_transaction(self, builder):
        """Test BUY transaction building."""
        tx = await builder.build_buy_transaction(
            token_address="0x1234...",
            amount=0.5,
            slippage=0.01
        )
        assert "sender" in tx
        assert "commands" in tx
        assert "gas" in tx

    @pytest.mark.asyncio
    async def test_build_sell_transaction(self, builder):
        """Test SELL transaction building."""
        tx = await builder.build_sell_transaction(
            token_address="0x1234...",
            token_object_id="0xabcd...",
            amount=100.0,
            slippage=0.02
        )
        assert "sender" in tx
        assert "commands" in tx


# ============================================================================
# Transaction Executor Tests
# ============================================================================

class TestTransactionExecutor:
    """Test transaction executor."""

    @pytest.fixture
    def executor(self):
        """Create test executor."""
        client = SuiGraphQLClient(network="testnet")
        wallet = Wallet.create_new()
        return TransactionExecutor(client, wallet, max_retries=2)

    def test_executor_initialization(self, executor):
        """Test executor initialization."""
        assert executor.max_retries == 2
        assert executor.retry_delay == 2.0
        assert executor.gas_multiplier == 1.2


# ============================================================================
# Trading Bot Tests
# ============================================================================

class TestTradingBot:
    """Test trading bot."""

    @pytest.fixture
    def bot(self):
        """Create test bot."""
        wallet = Wallet.create_new()
        return TradingBot(wallet, network="testnet")

    def test_bot_initialization(self, bot):
        """Test bot initialization."""
        assert bot.wallet is not None
        assert bot.network == "testnet"
        assert bot.client is not None

    @pytest.mark.asyncio
    async def test_check_balance(self, bot):
        """Test balance checking."""
        # Mock the balance query
        with patch.object(bot.client, 'get_balance', new_callable=AsyncMock) as mock:
            mock.return_value = 1_000_000_000
            balance = await bot.check_balance()
            assert balance == 1_000_000_000


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_full_trading_flow():
    """Test complete trading flow."""
    # Create wallet
    wallet = Wallet.create_new()

    # Create bot
    bot = TradingBot(wallet, network="testnet")

    # Create signal
    signal = TradingSignal(
        decision=TradingDecision.BUY,
        amount=0.1,
        confidence=0.85,
        token_address="0x1234...",
        metadata={"strategy": "test"}
    )

    # Verify signal properties
    assert signal.decision == TradingDecision.BUY
    assert bot.wallet.address == wallet.address


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
