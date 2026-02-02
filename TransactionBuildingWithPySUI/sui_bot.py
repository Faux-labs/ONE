import asyncio
import time
import json
from enum import Enum
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
import hashlib

from pysui import (
    SyncClient, SuiConfig, handle_result, SuiAddress,
    ObjectID, SuiU64, PysuiConfiguration, SyncGqlClient
)
from pysui.sui.sui_txn import SyncTransaction
from pysui.sui.sui_pgql import pgql_query as qn
from gql import gql

class TradeAction(Enum):
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class TradingDecision:
    action: TradeAction
    amount: float  # Amount of SUI or token
    confidence: float  # 0.0 to 1.0
    token_type: str = "0x2::sui::SUI"  # Default to SUI
    recipient: Optional[str] = None  # For SELL actions

class SuiGraphQLClient:
    def __init__(self, network: str = "testnet"):
        self.network = network
        # Initialize configuration for GraphQL
        try:
            cfg = PysuiConfiguration(group_name=PysuiConfiguration.SUI_GQL_RPC_GROUP)
            self.client = SyncGqlClient(pysui_config=cfg, write_schema=False)
        except Exception as e:
            print(f"Warning: Could not initialize GraphQL client: {e}")
            self.client = None
    
    def get_balance(self, owner_address: str, coin_type: str = "0x2::sui::SUI") -> int:
        """Get balance for specific coin type using GraphQL"""
        if not self.client:
            return 0
            
        query = """
        query GetBalance($owner: SuiAddress!, $coinType: String!) {
            address(address: $owner) {
                balance(type: $coinType) {
                    coinObjectCount
                    totalBalance
                }
            }
        }
        """
        variables = {"owner": owner_address, "coinType": coin_type}
        try:
            result = self.client.execute_query_string(string=query, encode_fn=lambda x: x)
            if result.is_ok():
                data = result.result_data
                if data and "data" in data and data["data"]["address"] and data["data"]["address"]["balance"]:
                    return int(data["data"]["address"]["balance"]["totalBalance"])
        except Exception as e:
            print(f"Error fetching balance: {e}")
        return 0
    
    def get_transaction_status(self, digest: str) -> Dict[str, Any]:
        """Check transaction status on-chain"""
        if not self.client:
            return None
            
        query = """
        query GetTransaction($digest: String!) {
            transactionBlock(digest: $digest) {
                effects {
                    status {
                        status
                    }
                }
            }
        }
        """
        try:
            result = self.client.execute_query_string(
                string=query, 
                encode_fn=lambda x: x,
                # Note: headers might need adjustment based on pysui version
            )
            return result.result_data if result.is_ok() else None
        except Exception as e:
            print(f"Error fetching transaction status: {e}")
            return None
    
    def get_reference_gas_price(self) -> int:
        """Get current epoch's reference gas price"""
        if not self.client:
            return 1000
            
        query = """
        query {
            epoch {
                referenceGasPrice
            }
        }
        """
        try:
            result = self.client.execute_query_string(string=query, encode_fn=lambda x: x)
            if result.is_ok():
                return int(result.result_data["data"]["epoch"]["referenceGasPrice"])
        except Exception as e:
            print(f"Error fetching gas price: {e}")
        return 1000  # Default fallback

class SuiWalletManager:
    def __init__(self, config_path: Optional[str] = None):
        try:
            self.config = SuiConfig.default_config() if not config_path else SuiConfig.user_config(config_path)
            self.client = SyncClient(self.config)
        except Exception as e:
            print(f"Warning: Could not load Sui config: {e}. Using mock/empty config for demonstration.")
            # In a real scenario, we'd need a valid config to interact with the chain
            self.config = None
            self.client = None
        self.graphql = SuiGraphQLClient()
    
    def create_wallet(self) -> Dict[str, Any]:
        """Create a new wallet (for development)"""
        from pysui import SuiKeyPair
        # Generate a new Ed25519 keypair
        keypair = SuiKeyPair.generate_new_keypair()
        address = keypair.to_address()
        
        return {
            "address": address.address,
            "private_key": keypair.export_to_bytes().hex(),
            "mnemonic": keypair.to_mnemonic()
        }
    
    def request_testnet_tokens(self, address: str) -> bool:
        """Request testnet SUI tokens (simulated airdrop)"""
        print(f"Use Sui faucet to fund address: {address}")
        print(f"Or transfer from pre-funded account in testnet")
        return True
    
    def get_all_balances(self, address: str) -> Dict[str, int]:
        """Get all token balances for an address"""
        if not self.graphql.client:
            return {}
            
        query = qn.GetAllCoinBalances(owner=SuiAddress(address))
        try:
            result = self.graphql.client.execute_query_node(with_node=query)
            balances = {}
            if result.is_ok():
                for balance in result.result_data.data:
                    balances[balance.coin_type] = int(balance.total_balance)
            return balances
        except Exception as e:
            print(f"Error fetching all balances: {e}")
            return {}

class GasManager:
    def __init__(self, graphql_client: SuiGraphQLClient):
        self.graphql = graphql_client
        
    def estimate_transaction_cost(self, 
                                 transaction: SyncTransaction,
                                 gas_price: Optional[int] = None) -> Dict[str, Any]:
        """Estimate gas cost for transaction"""
        # Use inspection to estimate computation units
        try:
            inspection = transaction.inspect_all()
            computation_units = self._estimate_computation_units(inspection)
            storage_units = self._estimate_storage_units(inspection)
        except Exception:
            # Fallback for demonstration if inspection fails
            computation_units = 1000
            storage_units = 5000
        
        # Get current reference gas price
        reference_price = self.graphql.get_reference_gas_price()
        current_price = gas_price if gas_price else reference_price
        
        storage_price = 100  # Fixed storage price
        
        total_cost = (computation_units * current_price) + (storage_units * storage_price)
        
        return {
            "computation_units": computation_units,
            "storage_units": storage_units,
            "reference_gas_price": reference_price,
            "suggested_gas_price": current_price,
            "estimated_cost": total_cost,
            "suggested_budget": int(total_cost * 1.2)  # 20% buffer
        }
    
    def _estimate_computation_units(self, inspection: Any) -> int:
        """Estimate computation units based on transaction complexity"""
        base_units = 1000  # Minimum bucket
        
        # Add complexity factors
        # Note: inspection structure depends on pysui version
        try:
            commands_count = len(getattr(inspection, 'commands', []))
        except:
            commands_count = 1
            
        complexity_score = commands_count * 500
        
        # Map to nearest bucket
        buckets = [1000, 5000, 10000, 20000, 50000, 200000, 1000000, 5000000]
        for bucket in buckets:
            if complexity_score <= bucket:
                return bucket
        return 5000000  # Max bucket
    
    def _estimate_storage_units(self, inspection: Any) -> int:
        """Estimate storage units (100 units per byte)"""
        try:
            object_count = len(getattr(inspection, 'objects_in_use', []))
        except:
            object_count = 1
        return object_count * 50 * 100
    
    def optimize_gas_price(self, 
                          urgency: float = 0.5,
                          network_congestion: float = 0.0) -> int:
        """Optimize gas price based on urgency and network conditions"""
        reference_price = self.graphql.get_reference_gas_price()
        
        # Add tip for urgency (0-100% of reference price)
        tip = int(reference_price * urgency * 0.1)  # Up to 10% tip
        
        # Adjust for congestion
        congestion_multiplier = 1.0 + (network_congestion * 0.5)  # Up to 50% increase
        
        optimized_price = int((reference_price + tip) * congestion_multiplier)
        
        return max(reference_price, optimized_price)

class SuiTradingBot:
    def __init__(self, wallet_manager: SuiWalletManager):
        self.wallet = wallet_manager
        self.gas_manager = GasManager(wallet_manager.graphql)
        self.retry_config = {
            'max_retries': 3,
            'base_delay': 1.0,
            'max_delay': 10.0
        }
    
    def execute_trade(self, decision: TradingDecision) -> Dict[str, Any]:
        """Main trading function with all requested features"""
        
        # 1. Validate decision
        if decision.confidence < 0.7:
            return {"status": "rejected", "reason": "low_confidence"}
        
        # 2. Check balance
        if not self.wallet.config:
            return {"status": "failed", "reason": "no_wallet_config"}
            
        sender_address = self.wallet.config.active_address
        balance = self.wallet.graphql.get_balance(sender_address, decision.token_type)
        
        if decision.action == TradeAction.SELL:
            if balance < decision.amount:
                return {"status": "failed", "reason": "insufficient_balance"}
        
        # 3. Build transaction
        try:
            transaction = self._build_transaction(decision)
        except Exception as e:
            return {"status": "failed", "reason": "build_error", "error": str(e)}
        
        # 4. Estimate and set gas budget
        gas_estimate = self.gas_manager.estimate_transaction_cost(transaction)
        gas_price = self.gas_manager.optimize_gas_price(
            urgency=decision.confidence,
            network_congestion=0.1
        )
        
        # 5. Execute with retry logic
        result = self._execute_with_retry(transaction, gas_estimate['suggested_budget'], gas_price)
        
        # 6. Verify on-chain
        if result.get('digest'):
            verification = self._verify_transaction(result['digest'])
            result.update(verification)
        
        return result
    
    def _build_transaction(self, decision: TradingDecision) -> SyncTransaction:
        """Build transaction based on trading decision"""
        if not self.wallet.client:
            raise Exception("Client not initialized")
            
        txn = SyncTransaction(client=self.wallet.client)
        
        if decision.action == TradeAction.BUY:
            # Placeholder for DEX logic
            txn.move_call(
                target="0x2::devnet_nft::mint", # Using a standard devnet call as example
                arguments=[
                    "Example NFT",
                    "Description",
                    "https://example.com/image.png"
                ]
            )
        else:  # SELL
            # For SELL: Transfer tokens
            coins = self.wallet.client.get_coins(
                owner=self.wallet.config.active_address,
                coin_type=decision.token_type
            )
            
            if coins.is_ok() and coins.result_data.data:
                coin_to_split = coins.result_data.data[0].coin_object_id
                
                # Split coin for the amount
                split_result = txn.split_coin(
                    coin=coin_to_split,
                    amounts=[int(decision.amount * 1e9)]
                )
                
                # Transfer to recipient
                txn.transfer_objects(
                    transfers=[split_result],
                    recipient=SuiAddress(decision.recipient) if decision.recipient 
                             else self.wallet.config.active_address
                )
            else:
                raise Exception("No coins found to sell")
        
        return txn
    
    def _execute_with_retry(self, 
                           transaction: SyncTransaction, 
                           gas_budget: int,
                           gas_price: int,
                           retry_count: int = 0) -> Dict[str, Any]:
        """Execute transaction with exponential backoff retry logic"""
        try:
            # Execute transaction
            result = transaction.execute(
                gas_budget=str(gas_budget),
                gas_price=str(gas_price)
            )
            
            if result.is_ok():
                tx_response = result.result_data
                return {
                    "status": "success",
                    "digest": tx_response.digest,
                    "gas_used": getattr(tx_response.effects.gas_used, 'total', 0),
                    "checkpoint": getattr(tx_response, 'checkpoint', None)
                }
            else:
                error_msg = result.result_string
                
                # Check if error is retryable
                if self._is_retryable_error(error_msg) and retry_count < self.retry_config['max_retries']:
                    delay = min(
                        self.retry_config['base_delay'] * (2 ** retry_count),
                        self.retry_config['max_delay']
                    )
                    
                    print(f"Retryable error: {error_msg}. Retrying in {delay}s...")
                    time.sleep(delay)
                    
                    return self._execute_with_retry(
                        transaction, gas_budget, gas_price, retry_count + 1
                    )
                else:
                    return {
                        "status": "failed",
                        "reason": "execution_error",
                        "error": error_msg
                    }
                    
        except Exception as e:
            if retry_count < self.retry_config['max_retries']:
                delay = min(
                    self.retry_config['base_delay'] * (2 ** retry_count),
                    self.retry_config['max_delay']
                )
                
                print(f"Exception: {str(e)}. Retrying in {delay}s...")
                time.sleep(delay)
                
                return self._execute_with_retry(
                    transaction, gas_budget, gas_price, retry_count + 1
                )
            else:
                return {
                    "status": "failed",
                    "reason": "max_retries_exceeded",
                    "error": str(e)
                }
    
    def _is_retryable_error(self, error_message: str) -> bool:
        """Determine if an error is retryable"""
        retryable_errors = [
            "timeout",
            "temporarily unavailable",
            "network error",
            "rate limit",
            "nonce too low",
            "gas too low"
        ]
        
        return any(err in error_message.lower() for err in retryable_errors)
    
    def _verify_transaction(self, digest: str) -> Dict[str, Any]:
        """Verify transaction success on-chain"""
        max_attempts = 5
        attempt = 0
        
        while attempt < max_attempts:
            tx_status = self.wallet.graphql.get_transaction_status(digest)
            
            if tx_status and tx_status.get("data", {}).get("transactionBlock"):
                status = tx_status["data"]["transactionBlock"]["effects"]["status"]["status"]
                
                if status == "success":
                    return {
                        "verified": True,
                        "on_chain_status": status,
                        "confirmation_attempts": attempt + 1
                    }
                elif status == "failure":
                    return {
                        "verified": False,
                        "on_chain_status": status,
                        "confirmation_attempts": attempt + 1
                    }
            
            attempt += 1
            time.sleep(1)
        
        return {
            "verified": False,
            "on_chain_status": "unknown",
            "confirmation_attempts": attempt
        }

if __name__ == "__main__":
    print("--- Sui Trading Bot Demonstration ---")
    
    # 1. Setup and Wallet Creation
    wallet_manager = SuiWalletManager()
    
    # Create new wallet
    new_wallet = wallet_manager.create_wallet()
    print(f"New wallet address: {new_wallet['address']}")
    
    # Request testnet tokens
    wallet_manager.request_testnet_tokens(new_wallet['address'])
    
    # Check balance
    balance = wallet_manager.graphql.get_balance(new_wallet['address'])
    print(f"Balance: {balance / 1e9} SUI")
    
    # 2. Execute a Trade (Simulation)
    bot = SuiTradingBot(wallet_manager)
    
    decision = TradingDecision(
        action=TradeAction.SELL,
        amount=0.1,
        confidence=0.85,
        token_type="0x2::sui::SUI"
    )
    
    print("\nExecuting trade...")
    result = bot.execute_trade(decision)
    print(f"Trade result: {json.dumps(result, indent=2)}")
    
    # 3. Gas Optimization
    print("\nGas Optimization:")
    ref_price = wallet_manager.graphql.get_reference_gas_price()
    print(f"Reference gas price: {ref_price}")
    
    urgent_price = bot.gas_manager.optimize_gas_price(urgency=0.9, network_congestion=0.3)
    print(f"Optimized gas price for urgent tx: {urgent_price}")
