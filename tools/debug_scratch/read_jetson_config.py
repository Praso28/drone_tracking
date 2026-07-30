import yaml

with open("config/jetson.yaml", "r") as f:
    cfg = yaml.safe_load(f)

print("=== Jetson Config File Contents ===")
print(cfg)
