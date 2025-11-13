import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Check what's in AASIST.py
import models.AASIST as aasist_module

print("AASIST module contents:")
for item in dir(aasist_module):
    if not item.startswith('_'):
        print(f"  {item}")

# Try to import specific classes
try:
    from models.AASIST import Model
    print("\nSuccessfully imported 'Model'")
except ImportError as e:
    print(f"\nCould not import 'Model': {e}")

try:
    from models.AASIST import AASIST
    print("Successfully imported 'AASIST'")
except ImportError as e:
    print(f"Could not import 'AASIST': {e}")

# Check if we can create an instance
try:
    model = aasist_module.Model()
    print("Successfully created Model instance")
except Exception as e:
    print(f"Could not create Model instance: {e}")