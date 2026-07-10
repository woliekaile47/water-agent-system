from pathlib import Path
import sys


SIMULATION_ROOT = Path(__file__).resolve().parents[1]
if str(SIMULATION_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMULATION_ROOT))
