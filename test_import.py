print("Testing corrected AASIST import...")

try:
    from models.AASIST import Model
    print("✅ SUCCESS: AASIST Model class imported correctly")
except ImportError as e:
    print(f"❌ FAILED: {e}")