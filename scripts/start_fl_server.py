# File: scripts/start_fl_server.py
# Federated Learning server startup script

#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.federated.fl_server import FederatedServer
from src.utils.logger import setup_logger
import logging

def main():
    """Start Federated Learning server"""
    
    parser = argparse.ArgumentParser(description="Start FE-AI Federated Learning Server")
    parser.add_argument("--model-type", default="gait_detector", 
                       choices=["gait_detector", "disease_classifier"],
                       help="Type of model for federated learning")
    parser.add_argument("--address", default="0.0.0.0:8080",
                       help="Server address and port")
    parser.add_argument("--rounds", type=int, default=50,
                       help="Number of federated learning rounds")
    parser.add_argument("--min-clients", type=int, default=3,
                       help="Minimum number of clients required")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("FLServer")
    logger.info(f"Starting Federated Learning server for {args.model_type}")
    
    try:
        # Initialize server
        fl_server = FederatedServer(
            model_type=args.model_type,
            server_address=args.address,
            num_rounds=args.rounds,
            min_clients=args.min_clients
        )
        
        # Start server (this will block)
        logger.info(f"FL Server starting on {args.address}")
        fl_server.start_server()
        
    except KeyboardInterrupt:
        logger.info("FL Server stopped by user")
    except Exception as e:
        logger.error(f"FL Server error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
