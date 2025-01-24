"""Logging configuration for the AML detection system."""
import logging

def setup_logger():
    """Configure and return the logger."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logger() 