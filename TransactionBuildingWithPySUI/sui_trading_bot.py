
"""
Sui Trading Bot with Transaction Building
==========================================

A comprehensive Python implementation for building and executing trading transactions
on the Sui blockchain using pysui and GraphQL.

Features:
- Wallet creation and management
- GraphQL client for Sui blockchain queries
- BUY/SELL transaction building
- Gas estimation and fee optimization
- Retry logic for failed transactions
- On-chain transaction verification
- Airdrop and balance checking
"""

import json
import time
import logging
import asyncio
from typing import Optional, Dict, Any, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import hashlib
import base64
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingDecision(Enum):
    """Trading decision types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradingSignal:
    """Trading signal data structure."""
    decision: TradingDecision
    amount: float  # Amount in base units
    confidence: float  # 0.0 to 1.0
    token_address: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GasConfig:
    """Gas configuration for transactions."""
    budget: int = 50_000_000  # Default gas budget in MIST
    price: Optional[int] = None  # Gas price (None for auto)
    payment: Optional[List[Dict]] = None  # Gas payment objects

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {"budget": self.budget}
        if self.price is not None:
            result["price"] = self.price
        if self.payment is not None:
            result["payment"] = self.payment
        return result


@dataclass
class TransactionResult:
    """Transaction execution result."""
    success: bool
    digest: Optional[str] = None
    effects: Optional[Dict] = None
    errors: List[str] = field(default_factory=list)
    gas_used: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "digest": self.digest,
            "effects": self.effects,
            "errors": self.errors,
            "gas_used": self.gas_used,
            "timestamp": self.timestamp.isoformat(),
            "retry_count": self.retry_count
        }


class Wallet:
    """Sui wallet management."""

    def __init__(self, address: str, private_key: Optional[str] = None, 
                 public_key: Optional[str] = None, mnemonic: Optional[str] = None):
        """Initialize wallet.

        Args:
            address: Sui address
            private_key: Private key (hex string)
            public_key: Public key (hex string)
            mnemonic: Recovery phrase
        """
        self.address = address
        self._private_key = private_key
        self._public_key = public_key
        self._mnemonic = mnemonic
        self._keypair = None

    @property
    def private_key(self) -> Optional[str]:
        """Get private key."""
        return self._private_key

    @property
    def public_key(self) -> Optional[str]:
        """Get public key."""
        return self._public_key

    @classmethod
    def create_new(cls) -> "Wallet":
        """Create a new wallet.

        Returns:
            New Wallet instance
        """
        try:
            # Try to use pysui for wallet creation
            from pysui.sui.sui_crypto import SuiKeyPair

            keypair = SuiKeyPair.generate()
            address = keypair.address
            private_key = keypair.private_key.hex() if hasattr(keypair, 'private_key') else None
            public_key = keypair.public_key.hex() if hasattr(keypair, 'public_key') else None

            wallet = cls(
                address=str(address),
                private_key=private_key,
                public_key=public_key
            )
            wallet._keypair = keypair
            logger.info(f"Created new wallet with address: {address}")
            return wallet

        except ImportError:
            logger.warning("pysui not available, using mock wallet creation")
            # Generate mock wallet for demonstration
            import secrets
            private_key = secrets.token_hex(32)
            public_key = hashlib.sha256(private_key.encode()).hexdigest()[:64]
            address = "0x" + hashlib.sha256(public_key.encode()).hexdigest()[:40]

            wallet = cls(
                address=address,
                private_key=private_key,
                public_key=public_key
            )
            logger.info(f"Created mock wallet with address: {address}")
            return wallet

    @classmethod
    def from_private_key(cls, private_key: str) -> "Wallet":
        """Create wallet from private key.

        Args:
            private_key: Private key hex string

        Returns:
            Wallet instance
        """
        try:
            from pysui.sui.sui_crypto import SuiKeyPair

            keypair = SuiKeyPair.from_bytes(bytes.fromhex(private_key))
            address = keypair.address
            public_key = keypair.public_key.hex() if hasattr(keypair, 'public_key') else None

            wallet = cls(
                address=str(address),
                private_key=private_key,
                public_key=public_key
            )
            wallet._keypair = keypair
            return wallet

        except ImportError:
            logger.warning("pysui not available, using mock wallet")
            public_key = hashlib.sha256(private_key.encode()).hexdigest()[:64]
            address = "0x" + hashlib.sha256(public_key.encode()).hexdigest()[:40]
            return cls(address=address, private_key=private_key, public_key=public_key)

    def sign_transaction(self, tx_bytes: bytes) -> str:
        """Sign transaction bytes.

        Args:
            tx_bytes: Transaction bytes to sign

        Returns:
            Signature string
        """
        if self._keypair:
            try:
                signature = self._keypair.sign(tx_bytes)
                return base64.b64encode(signature).decode()
            except Exception as e:
                logger.error(f"Error signing with keypair: {e}")

        # Fallback mock signature
        mock_sig = hashlib.sha256(tx_bytes + self._private_key.encode()).digest()
        return base64.b64encode(mock_sig).decode()


class SuiGraphQLClient:
    """GraphQL client for Sui blockchain."""

    # GraphQL endpoints for different networks
    ENDPOINTS = {
        "mainnet": "https://graphql.mainnet.sui.io/graphql",
        "testnet": "https://graphql.testnet.sui.io/graphql",
        "devnet": "https://graphql.devnet.sui.io/graphql"
    }

    def __init__(self, network: str = "testnet", endpoint: Optional[str] = None):
        """Initialize GraphQL client.

        Args:
            network: Network name (mainnet, testnet, devnet)
            endpoint: Custom endpoint URL (overrides network)
        """
        self.network = network
        self.endpoint = endpoint or self.ENDPOINTS.get(network, self.ENDPOINTS["testnet"])
        self._client = None
        self._session = None

    def _get_client(self):
        """Get or create GraphQL client."""
        if self._client is None:
            try:
                from gql import Client
                from gql.transport.aiohttp import AIOHTTPTransport

                transport = AIOHTTPTransport(url=self.endpoint)
                self._client = Client(transport=transport, fetch_schema_from_transport=True)
            except ImportError:
                logger.error("gql library not available")
                raise
        return self._client

    async def execute(self, query: str, variables: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute GraphQL query.

        Args:
            query: GraphQL query string
            variables: Query variables

        Returns:
            Query result dictionary
        """
        try:
            client = self._get_client()
            result = await client.execute_async(query, variable_values=variables)
            return result
        except Exception as e:
            logger.error(f"GraphQL query failed: {e}")
            raise

    def execute_sync(self, query: str, variables: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute GraphQL query synchronously.

        Args:
            query: GraphQL query string
            variables: Query variables

        Returns:
            Query result dictionary
        """
        try:
            from gql import Client
            from gql.transport.requests import RequestsHTTPTransport

            transport = RequestsHTTPTransport(url=self.endpoint)
            client = Client(transport=transport, fetch_schema_from_transport=False)
            result = client.execute(query, variable_values=variables)
            return result
        except Exception as e:
            logger.error(f"GraphQL query failed: {e}")
            raise

    async def get_balance(self, address: str, coin_type: str = "0x2::sui::SUI") -> int:
        """Get coin balance for address.

        Args:
            address: Sui address
            coin_type: Coin type (default: SUI)

        Returns:
            Balance in MIST
        """
        query = """
        query GetBalance($address: SuiAddress!, $coinType: String!) {
            address(address: $address) {
                balance(type: $coinType) {
                    totalBalance
                }
            }
        }
        """
        variables = {"address": address, "coinType": coin_type}

        try:
            result = await self.execute(query, variables)
            balance_str = result.get("address", {}).get("balance", {}).get("totalBalance", "0")
            return int(balance_str)
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0

    def get_balance_sync(self, address: str, coin_type: str = "0x2::sui::SUI") -> int:
        """Get coin balance synchronously.

        Args:
            address: Sui address
            coin_type: Coin type (default: SUI)

        Returns:
            Balance in MIST
        """
        query = """
        query GetBalance($address: SuiAddress!, $coinType: String!) {
            address(address: $address) {
                balance(type: $coinType) {
                    totalBalance
                }
            }
        }
        """
        variables = {"address": address, "coinType": coin_type}

        try:
            result = self.execute_sync(query, variables)
            balance_str = result.get("address", {}).get("balance", {}).get("totalBalance", "0")
            return int(balance_str)
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0

    async def get_coins(self, address: str, coin_type: str = "0x2::sui::SUI") -> List[Dict]:
        """Get coin objects for address.

        Args:
            address: Sui address
            coin_type: Coin type

        Returns:
            List of coin objects
        """
        query = """
        query GetCoins($address: SuiAddress!, $coinType: String!) {
            address(address: $address) {
                coins(type: $coinType, first: 50) {
                    nodes {
                        coinObjectId
                        balance
                    }
                }
            }
        }
        """
        variables = {"address": address, "coinType": coin_type}

        try:
            result = await self.execute(query, variables)
            coins = result.get("address", {}).get("coins", {}).get("nodes", [])
            return coins
        except Exception as e:
            logger.error(f"Failed to get coins: {e}")
            return []

    async def get_transaction(self, digest: str) -> Optional[Dict]:
        """Get transaction by digest.

        Args:
            digest: Transaction digest

        Returns:
            Transaction data or None
        """
        query = """
        query GetTransaction($digest: String!) {
            transactionBlock(digest: $digest) {
                digest
                sender {
                    address
                }
                effects {
                    status
                    gasUsed {
                        computationCost
                        storageCost
                        storageRebate
                    }
                    timestamp
                }
            }
        }
        """
        variables = {"digest": digest}

        try:
            result = await self.execute(query, variables)
            return result.get("transactionBlock")
        except Exception as e:
            logger.error(f"Failed to get transaction: {e}")
            return None

    async def request_airdrop(self, address: str, amount: int = 1_000_000_000) -> bool:
        """Request SUI airdrop from faucet (devnet/testnet only).

        Args:
            address: Sui address
            amount: Amount to request (default: 1 SUI)

        Returns:
            True if successful
        """
        if self.network == "mainnet":
            logger.error("Airdrop not available on mainnet")
            return False

        faucet_urls = {
            "devnet": "https://faucet.devnet.sui.io/gas",
            "testnet": "https://faucet.testnet.sui.io/gas"
        }

        faucet_url = faucet_urls.get(self.network)
        if not faucet_url:
            logger.error(f"No faucet available for {self.network}")
            return False

        try:
            import aiohttp

            payload = {
                "FixedAmountRequest": {
                    "recipient": address
                }
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(faucet_url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Airdrop requested: {result}")
                        return True
                    else:
                        logger.error(f"Airdrop failed: {response.status}")
                        return False

        except Exception as e:
            logger.error(f"Airdrop request failed: {e}")
            return False

    def request_airdrop_sync(self, address: str, amount: int = 1_000_000_000) -> bool:
        """Request SUI airdrop synchronously.

        Args:
            address: Sui address
            amount: Amount to request

        Returns:
            True if successful
        """
        if self.network == "mainnet":
            logger.error("Airdrop not available on mainnet")
            return False

        faucet_urls = {
            "devnet": "https://faucet.devnet.sui.io/gas",
            "testnet": "https://faucet.testnet.sui.io/gas"
        }

        faucet_url = faucet_urls.get(self.network)
        if not faucet_url:
            logger.error(f"No faucet available for {self.network}")
            return False

        try:
            import requests

            payload = {
                "FixedAmountRequest": {
                    "recipient": address
                }
            }

            response = requests.post(faucet_url, json=payload)
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Airdrop requested: {result}")
                return True
            else:
                logger.error(f"Airdrop failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Airdrop request failed: {e}")
            return False


class TransactionBuilder:
    """Builder for Sui transactions."""

    def __init__(self, client: SuiGraphQLClient, wallet: Wallet):
        """Initialize transaction builder.

        Args:
            client: GraphQL client
            wallet: Wallet for signing
        """
        self.client = client
        self.wallet = wallet
        self._transaction = None

    async def build_buy_transaction(
        self,
        token_address: str,
        amount: float,
        slippage: float = 0.01,
        gas_config: Optional[GasConfig] = None
    ) -> Dict[str, Any]:
        """Build BUY transaction.

        Args:
            token_address: Token package address
            amount: Amount to buy (in SUI)
            slippage: Slippage tolerance (default: 1%)
            gas_config: Gas configuration

        Returns:
            Transaction dictionary
        """
        gas_config = gas_config or GasConfig()
        amount_mist = int(amount * 1_000_000_000)  # Convert SUI to MIST

        logger.info(f"Building BUY transaction: {amount} SUI -> {token_address}")

        # This would typically call a DEX move function
        # For demonstration, we build a split_coin + move_call transaction
        tx_data = {
            "sender": self.wallet.address,
            "commands": [
                {
                    "SplitCoins": {
                        "coin": "GasCoin",
                        "amounts": [amount_mist]
                    }
                },
                {
                    "MoveCall": {
                        "package": token_address,
                        "module": "marketplace",
                        "function": "buy",
                        "arguments": ["Result(0)", int(slippage * 10000)],  # 1% = 100 bps
                        "type_arguments": []
                    }
                }
            ],
            "gas": gas_config.to_dict()
        }

        return tx_data

    async def build_sell_transaction(
        self,
        token_address: str,
        token_object_id: str,
        amount: float,
        slippage: float = 0.01,
        gas_config: Optional[GasConfig] = None
    ) -> Dict[str, Any]:
        """Build SELL transaction.

        Args:
            token_address: Token package address
            token_object_id: Token object to sell
            amount: Amount to sell
            slippage: Slippage tolerance
            gas_config: Gas configuration

        Returns:
            Transaction dictionary
        """
        gas_config = gas_config or GasConfig()
        amount_units = int(amount * 1_000_000_000)  # Adjust based on token decimals

        logger.info(f"Building SELL transaction: {amount} tokens -> SUI")

        tx_data = {
            "sender": self.wallet.address,
            "commands": [
                {
                    "MoveCall": {
                        "package": token_address,
                        "module": "marketplace",
                        "function": "sell",
                        "arguments": [token_object_id, amount_units, int(slippage * 10000)],
                        "type_arguments": []
                    }
                }
            ],
            "gas": gas_config.to_dict()
        }

        return tx_data

    async def estimate_gas(self, tx_data: Dict[str, Any]) -> int:
        """Estimate gas for transaction.

        Args:
            tx_data: Transaction data

        Returns:
            Estimated gas cost in MIST
        """
        try:
            # Use dry run to estimate gas
            query = """
            mutation DryRunTransactionBlock($txBytes: String!) {
                dryRunTransactionBlock(txBytes: $txBytes) {
                    effects {
                        gasUsed {
                            computationCost
                            storageCost
                            storageRebate
                        }
                    }
                    error
                }
            }
            """

            # Serialize transaction for dry run
            tx_bytes = base64.b64encode(json.dumps(tx_data).encode()).decode()
            variables = {"txBytes": tx_bytes}

            result = await self.client.execute(query, variables)
            dry_run = result.get("dryRunTransactionBlock", {})

            if dry_run.get("error"):
                logger.warning(f"Dry run error: {dry_run['error']}")
                return 50_000_000  # Default fallback

            gas_used = dry_run.get("effects", {}).get("gasUsed", {})
            computation = int(gas_used.get("computationCost", 0))
            storage = int(gas_used.get("storageCost", 0))
            rebate = int(gas_used.get("storageRebate", 0))

            total = computation + storage - rebate
            logger.info(f"Estimated gas: {total} MIST")
            return max(total, 1_000_000)  # Minimum 1M MIST

        except Exception as e:
            logger.warning(f"Gas estimation failed: {e}, using default")
            return 50_000_000  # Default 0.05 SUI


class TransactionExecutor:
    """Executes transactions with retry logic."""

    def __init__(
        self,
        client: SuiGraphQLClient,
        wallet: Wallet,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        gas_multiplier: float = 1.2
    ):
        """Initialize transaction executor.

        Args:
            client: GraphQL client
            wallet: Wallet for signing
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries (seconds)
            gas_multiplier: Gas budget multiplier on retry
        """
        self.client = client
        self.wallet = wallet
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.gas_multiplier = gas_multiplier

    async def execute(
        self,
        tx_data: Dict[str, Any],
        verify: bool = True,
        timeout: float = 60.0
    ) -> TransactionResult:
        """Execute transaction with retry logic.

        Args:
            tx_data: Transaction data
            verify: Whether to verify on-chain
            timeout: Execution timeout

        Returns:
            Transaction result
        """
        retry_count = 0
        last_error = None
        current_gas_budget = tx_data.get("gas", {}).get("budget", 50_000_000)

        while retry_count <= self.max_retries:
            try:
                # Update gas budget for retries
                if retry_count > 0:
                    current_gas_budget = int(current_gas_budget * self.gas_multiplier)
                    tx_data["gas"]["budget"] = current_gas_budget
                    logger.info(f"Retry {retry_count}: Increased gas budget to {current_gas_budget}")

                # Build and sign transaction
                tx_bytes = await self._serialize_transaction(tx_data)
                signature = self.wallet.sign_transaction(tx_bytes)

                # Execute transaction
                result = await self._submit_transaction(tx_bytes, signature)

                if result.get("errors"):
                    raise Exception(f"Transaction failed: {result['errors']}")

                digest = result.get("digest")

                # Verify on-chain if requested
                if verify and digest:
                    verified = await self._verify_transaction(digest, timeout)
                    if not verified:
                        raise Exception("Transaction verification failed")

                logger.info(f"Transaction executed successfully: {digest}")

                return TransactionResult(
                    success=True,
                    digest=digest,
                    effects=result.get("effects"),
                    gas_used=result.get("gas_used"),
                    retry_count=retry_count
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Transaction attempt {retry_count + 1} failed: {e}")
                retry_count += 1

                if retry_count <= self.max_retries:
                    await asyncio.sleep(self.retry_delay * retry_count)

        logger.error(f"Transaction failed after {self.max_retries + 1} attempts")
        return TransactionResult(
            success=False,
            errors=[last_error] if last_error else ["Unknown error"],
            retry_count=retry_count
        )

    async def _serialize_transaction(self, tx_data: Dict[str, Any]) -> bytes:
        """Serialize transaction data to bytes.

        Args:
            tx_data: Transaction data

        Returns:
            Serialized transaction bytes
        """
        # In production, this would use proper BCS serialization
        # For now, we use a JSON-based approach
        return json.dumps(tx_data, sort_keys=True).encode()

    async def _submit_transaction(self, tx_bytes: bytes, signature: str) -> Dict[str, Any]:
        """Submit transaction to network.

        Args:
            tx_bytes: Transaction bytes
            signature: Transaction signature

        Returns:
            Submission result
        """
        query = """
        mutation ExecuteTransactionBlock(
            $txBytes: String!
            $signatures: [String!]!
        ) {
            executeTransactionBlock(
                txBytes: $txBytes
                signatures: $signatures
            ) {
                digest
                effects {
                    status
                    gasUsed {
                        computationCost
                        storageCost
                        storageRebate
                    }
                }
                errors {
                    message
                }
            }
        }
        """

        variables = {
            "txBytes": base64.b64encode(tx_bytes).decode(),
            "signatures": [signature]
        }

        result = await self.client.execute(query, variables)
        execution = result.get("executeTransactionBlock", {})

        return {
            "digest": execution.get("digest"),
            "effects": execution.get("effects"),
            "errors": execution.get("errors"),
            "gas_used": self._calculate_gas_used(execution.get("effects", {}).get("gasUsed", {}))
        }

    def _calculate_gas_used(self, gas_data: Dict) -> int:
        """Calculate total gas used.

        Args:
            gas_data: Gas usage data

        Returns:
            Total gas in MIST
        """
        computation = int(gas_data.get("computationCost", 0))
        storage = int(gas_data.get("storageCost", 0))
        rebate = int(gas_data.get("storageRebate", 0))
        return computation + storage - rebate

    async def _verify_transaction(self, digest: str, timeout: float) -> bool:
        """Verify transaction on-chain.

        Args:
            digest: Transaction digest
            timeout: Verification timeout

        Returns:
            True if verified
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                tx = await self.client.get_transaction(digest)
                if tx and tx.get("effects", {}).get("status") == "SUCCESS":
                    logger.info(f"Transaction {digest} verified on-chain")
                    return True
                elif tx and tx.get("effects", {}).get("status") == "FAILURE":
                    logger.error(f"Transaction {digest} failed on-chain")
                    return False

            except Exception as e:
                logger.debug(f"Verification check failed: {e}")

            await asyncio.sleep(2)

        logger.warning(f"Transaction verification timed out: {digest}")
        return False


class TradingBot:
    """Main trading bot class."""

    def __init__(
        self,
        wallet: Wallet,
        network: str = "testnet",
        graphql_endpoint: Optional[str] = None,
        max_retries: int = 3
    ):
        """Initialize trading bot.

        Args:
            wallet: Trading wallet
            network: Network name
            graphql_endpoint: Custom GraphQL endpoint
            max_retries: Max transaction retries
        """
        self.wallet = wallet
        self.client = SuiGraphQLClient(network, graphql_endpoint)
        self.tx_builder = TransactionBuilder(self.client, wallet)
        self.tx_executor = TransactionExecutor(
            self.client, 
            wallet, 
            max_retries=max_retries
        )
        self.network = network

        logger.info(f"Trading bot initialized on {network} with wallet {wallet.address}")

    async def check_balance(self, coin_type: str = "0x2::sui::SUI") -> int:
        """Check wallet balance.

        Args:
            coin_type: Coin type

        Returns:
            Balance in MIST
        """
        balance = await self.client.get_balance(self.wallet.address, coin_type)
        logger.info(f"Balance: {balance / 1_000_000_000:.9f} SUI")
        return balance

    def check_balance_sync(self, coin_type: str = "0x2::sui::SUI") -> int:
        """Check wallet balance synchronously.

        Args:
            coin_type: Coin type

        Returns:
            Balance in MIST
        """
        balance = self.client.get_balance_sync(self.wallet.address, coin_type)
        logger.info(f"Balance: {balance / 1_000_000_000:.9f} SUI")
        return balance

    async def request_airdrop(self, amount: int = 1_000_000_000) -> bool:
        """Request SUI airdrop.

        Args:
            amount: Amount to request

        Returns:
            True if successful
        """
        return await self.client.request_airdrop(self.wallet.address, amount)

    def request_airdrop_sync(self, amount: int = 1_000_000_000) -> bool:
        """Request SUI airdrop synchronously.

        Args:
            amount: Amount to request

        Returns:
            True if successful
        """
        return self.client.request_airdrop_sync(self.wallet.address, amount)

    async def execute_trade(
        self,
        signal: TradingSignal,
        slippage: float = 0.01,
        gas_config: Optional[GasConfig] = None,
        verify: bool = True
    ) -> TransactionResult:
        """Execute a trade based on trading signal.

        Args:
            signal: Trading signal
            slippage: Slippage tolerance
            gas_config: Gas configuration
            verify: Verify on-chain

        Returns:
            Transaction result
        """
        if signal.decision == TradingDecision.HOLD:
            logger.info("HOLD signal - no action taken")
            return TransactionResult(success=True, errors=[])

        # Build transaction
        if signal.decision == TradingDecision.BUY:
            tx_data = await self.tx_builder.build_buy_transaction(
                token_address=signal.token_address,
                amount=signal.amount,
                slippage=slippage,
                gas_config=gas_config
            )
        else:  # SELL
            # For SELL, we need the token object ID
            token_object_id = signal.metadata.get("token_object_id")
            if not token_object_id:
                return TransactionResult(
                    success=False,
                    errors=["SELL requires token_object_id in metadata"]
                )

            tx_data = await self.tx_builder.build_sell_transaction(
                token_address=signal.token_address,
                token_object_id=token_object_id,
                amount=signal.amount,
                slippage=slippage,
                gas_config=gas_config
            )

        # Estimate gas
        estimated_gas = await self.tx_builder.estimate_gas(tx_data)
        logger.info(f"Estimated gas: {estimated_gas} MIST")

        # Execute with retry
        result = await self.tx_executor.execute(tx_data, verify=verify)

        if result.success:
            logger.info(f"Trade executed: {signal.decision.value} {signal.amount}")
        else:
            logger.error(f"Trade failed: {result.errors}")

        return result

    async def get_optimal_gas_config(self) -> GasConfig:
        """Get optimal gas configuration based on network conditions.

        Returns:
            Optimized gas config
        """
        try:
            # Query current gas price
            query = """
            query GetGasPrice {
                epoch {
                    referenceGasPrice
                }
            }
            """

            result = await self.client.execute(query)
            gas_price = int(result.get("epoch", {}).get("referenceGasPrice", 1000))

            # Add buffer for faster confirmation
            optimal_price = int(gas_price * 1.1)

            logger.info(f"Optimal gas price: {optimal_price}")
            return GasConfig(price=optimal_price, budget=50_000_000)

        except Exception as e:
            logger.warning(f"Failed to get optimal gas: {e}")
            return GasConfig()


# Convenience functions for quick usage

def create_wallet() -> Wallet:
    """Create a new wallet.

    Returns:
        New wallet
    """
    return Wallet.create_new()


def load_wallet(private_key: str) -> Wallet:
    """Load wallet from private key.

    Args:
        private_key: Private key hex

    Returns:
        Wallet instance
    """
    return Wallet.from_private_key(private_key)


async def create_trading_bot(
    private_key: Optional[str] = None,
    network: str = "testnet"
) -> TradingBot:
    """Create trading bot with optional private key.

    Args:
        private_key: Optional private key (creates new wallet if None)
        network: Network name

    Returns:
        TradingBot instance
    """
    if private_key:
        wallet = load_wallet(private_key)
    else:
        wallet = create_wallet()

    return TradingBot(wallet, network)


# Example usage
async def main():
    """Example usage of the trading bot."""

    # Create a new wallet
    wallet = create_wallet()
    print(f"Created wallet: {wallet.address}")

    # Initialize trading bot on testnet
    bot = TradingBot(wallet, network="testnet")

    # Request airdrop for testing
    print("Requesting airdrop...")
    airdrop_success = await bot.request_airdrop()
    print(f"Airdrop success: {airdrop_success}")

    # Check balance
    balance = await bot.check_balance()
    print(f"Balance: {balance / 1_000_000_000} SUI")

    # Create a trading signal
    signal = TradingSignal(
        decision=TradingDecision.BUY,
        amount=0.1,  # 0.1 SUI
        confidence=0.85,
        token_address="0x1234...",  # Token package address
        metadata={"strategy": "momentum"}
    )

    # Get optimal gas config
    gas_config = await bot.get_optimal_gas_config()

    # Execute trade
    result = await bot.execute_trade(
        signal=signal,
        slippage=0.01,
        gas_config=gas_config,
        verify=True
    )

    print(f"Trade result: {result.to_dict()}")


if __name__ == "__main__":
    asyncio.run(main())
