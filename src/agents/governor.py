import os
import logging
from src.core.state import GraphState

logger = logging.getLogger(__name__)

def governor(state: GraphState) -> str:
    """Circuit breaker for infinite loop prevention."""
    iteration_count = state.get("iteration_count", 0)
    max_iterations = int(os.getenv("MAX_ITERATIONS", "5"))
    
    logger.info(f"Governor Check: iteration {iteration_count} / {max_iterations}")
    
    if iteration_count >= max_iterations:
        logger.warning(f"Governor triggered: Max iterations ({max_iterations}) reached. Force routing to finalize.")
        return "finalize"
        
    return "critic_or_next_step"
