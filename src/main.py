#!/usr/bin/env python3
"""
main.py

Entry point for the open_perception. Loads configuration,
initializes modules, and runs the main pipeline loop.
"""

import argparse
import sys
import os
from pprint import pprint

# Example imports (adjust to match your actual file structure)
from open_perception.utils.config_loader import load_config
from open_perception.pipeline.orchestrator import Orchestrator
# from open_perception.pipeline.tracker import Tracker
# from open_perception.pipeline.aggregator import Aggregator
# from open_perception.communication.api_server import APIServer
from open_perception.communication.redis_interface import RedisInterface
# from open_perception.communication.ros_interface import ROSInterface
from open_perception.communication.gui_interface import GUIInterface


def parse_arguments():
    """
    Parse command-line arguments to specify a configuration file and other options.
    """
    parser = argparse.ArgumentParser(
        description="Open Vocabulary Perception Pipeline"
    )
    script_path = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(script_path, "../config/default.yaml"),
        help="Path to the configuration file (YAML)."
    )
    return parser.parse_args()

def initialize_communication(config):
    """
    Initialize communication interfaces based on the config.

    Returns a dictionary of enabled communication modules, e.g.:
    {
        'api': APIServer(...),
        'redis': RedisInterface(...),
        'ros': ROSInterface(...)
    }
    """
    communication_modules = {}

    # if config.get("communication", {}).get("api_server", {}).get("enabled", False):
    #     api_config = config["communication"]["api_server"]
    #     communication_modules['api'] = APIServer(
    #         host=api_config["host"],
    #         port=api_config["port"]
    #     )

    if config.get("communication", {}).get("redis", {}).get("enabled", False):
        redis_config = config["communication"]["redis"]
        communication_modules['redis'] = RedisInterface(
            config=redis_config
        )

    # if config.get("communication", {}).get("ros", {}).get("enabled", False):
    #     ros_config = config["communication"]["ros"]
    #     communication_modules['ros'] = ROSInterface(
    #         node_name=ros_config.get("node_name", "open_vocab_node")
    #     )
    
    if config.get("communication", {}).get("gui", {}).get("enabled", False):
        gui_config = config["communication"]["gui"]
        communication_modules['gui'] = GUIInterface(
            config=gui_config, models_config=config.get("pipeline",{}).get("perception", {}).get("models", {})
        )

    return communication_modules

def main():
    """
    Main function that coordinates the setup and execution of the pipeline.
    """
    # 1. Parse command-line arguments
    args = parse_arguments()

    # 2. Load configuration
    try:
        print(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        pprint(config)
    except FileNotFoundError:
        print(f"Error: Configuration file {args.config} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # 3. Initialize communication modules (if enabled)
    comm_modules = initialize_communication(config)

    # 4. Initialize aggregator and tracker (these might be optional or configured differently)
    tracker = None
    if config.get("pipeline", {}).get("tracking", {}).get("enabled", False):
        tracker_config = config["pipeline"]["tracking"]
        tracker = Tracker(tracker_config)

    aggregator = None
    if config.get("pipeline", {}).get("aggregation", {}).get("enabled", False):
        aggregator_config = config["pipeline"]["aggregation"]
        aggregator = Aggregator(aggregator_config)

    # 5. Create the Orchestrator and pass in the relevant components
    orchestrator = Orchestrator(
        config=config,
        tracker=tracker,
        aggregator=aggregator,
        comm_modules=comm_modules
    )

    # 6. Start communication services (e.g., run a server, subscribe to channels, etc.)
    #    Depending on how you implement communication modules, you might start them here.
    for module_name, module in comm_modules.items():
        if hasattr(module, "start"):
            module.start()

    # 7. Run the main pipeline loop
    #    The orchestrator might pull frames from a camera, run inference using
    #    multiple perception models, track objects, and publish results.
    
    print("Pipeline started successfully.")
    
    # def orchestrator_thread():
    #     orchestrator.run()
    try:
        orchestrator.run()
        # orchestrator_thread = threading.Thread(target=orchestrator_thread)
        # orchestrator_thread.start()
        # while orchestrator_thread.is_alive():
        #     time.sleep(1)
        #     print("Pipeline is running...")
        # orchestrator.run()
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
    finally:
        # 8. Clean up or gracefully shut down
        for module_name, module in comm_modules.items():
            if hasattr(module, "shutdown"):
                module.shutdown()

        print("Pipeline shut down successfully.")

if __name__ == "__main__":
    print("Starting Open Vocabulary Perception Pipeline...")
    main()
