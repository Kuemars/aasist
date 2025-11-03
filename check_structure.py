print("Checking AASIST model structure...")

# First, let's see what's in the AASIST module
try:
    from models import AASIST
    print("✅ AASIST module imported successfully")
    print("Available items in AASIST module:")
    for item in dir(AASIST):
        if not item.startswith('_'):
            print(f"  - {item}")
except ImportError as e:
    print(f"❌ Failed to import AASIST: {e}")

# Also check the models directory
print("\nChecking models directory:")
import os
model_files = os.listdir('models')
for file in model_files:
    if file.endswith('.py'):
        print(f"  - {file}")