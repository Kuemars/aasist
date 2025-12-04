# scripts/check_aasist_config.py
import os
import sys
import yaml

# Get project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Check config file
config_path = os.path.join(PROJECT_ROOT, "config", "AASIST.conf")
print(f"🔍 Checking config file: {config_path}")

if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    print("✅ Config file exists")
    print(f"\n📄 Config content:")
    
    if 'model_config' in config:
        model_config = config['model_config']
        print("Model Config:")
        for key, value in model_config.items():
            print(f"  {key}: {value}")
    else:
        print("Full Config:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    # Check for required keys
    required_keys = ['filts', 'nb_samp', 'nb_classes']
    print(f"\n🔑 Required keys check:")
    for key in required_keys:
        if key in config.get('model_config', config):
            print(f"  ✅ {key}: {config.get('model_config', config).get(key)}")
        else:
            print(f"  ❌ {key}: MISSING")
else:
    print("❌ Config file not found")
    
    # Show what's in the config directory
    config_dir = os.path.join(PROJECT_ROOT, "config")
    if os.path.exists(config_dir):
        print(f"\n📁 Contents of config directory:")
        for item in os.listdir(config_dir):
            print(f"  {item}")
    else:
        print(f"\n📁 config directory doesn't exist")

# Try to import model with minimal config
print(f"\n🧪 Testing model creation with minimal config...")

try:
    from models.AASIST import Model
    
    # Minimal config that should work
    test_config = {
        'filts': [80, [1, 32], [32, 32], [32, 64], [64, 64]],
        'nb_samp': 64600,
        'nb_classes': 2,
        'first_conv': 1024,
        'gru_node': 1024,
        'nb_gru_layer': 3,
    }
    
    model = Model(test_config)
    print("✅ Model created successfully with test config")
    
except Exception as e:
    print(f"❌ Error creating model: {e}")