import argparse
from hat.mapping import station_mapping
import json

def main():
    parser = argparse.ArgumentParser(
        description="Station mapping tool: maps stations on provided grid."
    )
    parser.add_argument(
        "config_file", type=str, help="Path to the JSON configuration file"
    )
    args = parser.parse_args()
    # Load configuration from the specified JSON file
    with open(args.config_file, "r") as file:
        config = json.load(file)
    station_mapping(config)
    
if __name__ == "__main__":
    main()