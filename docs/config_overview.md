
## Configuration

All pipeline settings reside in **YAML** files under the `config/` directory. The main configuration file is `default.yaml`, which defines:

- **Model**: Which perception model(s) to use and their paths/checkpoints.  
- **Communication**: Whether to enable the API server, Redis, or ROS, plus connection details.  
- **Detection Parameters**: Confidence thresholds, image size, etc.  
- **Tracking Parameters**: Settings for object tracking across frames.

You can override defaults by specifying another config file or passing command-line arguments. For example:

```bash
python src/main.py --config config/multigranular_dino_sam2.yaml
```

---

## Communication Interfaces

### Redis

1. **Enable**: Set `communication.redis.enabled: true`.  
2. **Channels**: Subscribes to a configured channel for input frames (or metadata) and publishes detection/tracking results.  
3. **Implementation**: Check `src/communication/redis_interface.py` for details.

### API (API/REST)

1. **Enable**: Set `communication.api_server.enabled: true` in the config file.  
2. **Run**: Access endpoints (e.g., `GET /detections`) on the specified host/port.  
3. **Configuration**: Located in `config/default.yaml` or overridden in your custom config.

### ROS

1. **Enable**: Set `communication.ros.enabled: true`.  
2. **Topics**: Publishes bounding boxes, detection messages, and optionally subscribes to an image topic.  
3. **Implementation**: Check `src/communication/ros_interface.py` for details.