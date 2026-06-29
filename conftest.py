import sys
from pathlib import Path

# Make the project root importable so `from src.data.card_parser import ...` works
# without requiring a pip-installed package.
sys.path.insert(0, str(Path(__file__).parent))
