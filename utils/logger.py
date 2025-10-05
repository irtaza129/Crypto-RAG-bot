# Basic logger setup for the project
import logging

logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
	handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("chatbot")
