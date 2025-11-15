"""
test_api_interface_and_client.py
"""
# %%
import pytest
import numpy as np

from open_perception.communication.api_interface import APIInterface
from open_perception.communication.api_client import APIClient
from open_perception.utils.config_loader import load_config
import os
import time

# load API config from default config file
config = load_config()
api_config = config["communication"]["api"]

interface = APIInterface(config=api_config)
client = APIClient(config=api_config)
#%%

@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    print("Starting API interface and client...")
    interface.start()
    client.start()
    api_delay()
    yield
    print("Stopping API interface and client...")
    interface.stop()
    client.stop()

def api_delay():
    # delay test execution to allow API server to process messages
    time.sleep(0)

def test_reset():
    # Placeholder for reset test
    pass

def test_locate_signal():
    obj_name = "object_1"
    img = np.zeros((480, 640, 3), dtype=np.uint8)  # Dummy image

    # test if object names are requested to be located
    elements = client.locate(obj_name, img, wait=False)
    api_delay()
    # Placeholder for checking if object is located
    pass

def test_remove_signal():
    obj_name = "object_1"
    # test if object names are requested to be removed
    client.remove(obj_name)
    api_delay()
    # Placeholder for checking if object is removed
    pass

def test_update_config():
    new_config = {"logging": {"level": "DEBUG", "format": "%(asctime)s - %(message)s"}}
    client.update_config(new_config)
    api_delay()
    # Placeholder for checking if config is updated
    pass

def test_frames():
    sensor_name = "front"
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    meta = {"stamp": time.time(), "id": 1}

    # send rgb frame and check if it is received
    client.send_frame(sensor_name, rgb=rgb, meta=meta, wait=True, fake_pc=False)
    api_delay()

    rgb_frames = interface.get_frame()
    assert sensor_name in rgb_frames
    assert "rgb" in rgb_frames[sensor_name]
    assert np.array_equal(rgb_frames[sensor_name]["rgb"], rgb)

    # check if synced frames are empty before sending point cloud
    frames = interface.get_synced_frame_and_pc()
    assert frames == {} or frames is None

    # send point cloud and check if it is received
    point_cloud = np.random.rand(480, 640, 3)
    client.send_frame(sensor_name, point_cloud=point_cloud, meta=meta, wait=True, fake_pc=False)
    api_delay()
    pc = interface.get_point_cloud()
    assert sensor_name in pc
    assert "point_cloud" in pc[sensor_name]
    assert np.array_equal(pc[sensor_name]["point_cloud"], point_cloud)
    assert pc[sensor_name]["index"] == 1

    # check if the frames are now in sync
    frames = interface.get_synced_frame_and_pc()
    assert sensor_name in frames
    assert "rgb" in frames[sensor_name]    
    assert "point_cloud" in frames[sensor_name]
    assert np.array_equal(frames[sensor_name]["rgb"], rgb)
    assert np.array_equal(frames[sensor_name]["point_cloud"], point_cloud)

    # update meta and check if the synced frames are still the previous ones
    meta = {"stamp": time.time(), "id": 2}
    new_rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    client.send_frame(sensor_name, rgb=new_rgb, meta=meta, wait=True, fake_pc=False)
    api_delay()
    assert np.array_equal(frames[sensor_name]["rgb"], rgb)
    assert np.array_equal(frames[sensor_name]["point_cloud"], point_cloud)
    assert frames[sensor_name]["index"] == 1
    
    # check if the new synced frames are received
    client.send_frame(sensor_name, point_cloud=point_cloud, meta=meta, wait=True, fake_pc=False)
    api_delay()
    frames = interface.get_synced_frame_and_pc()
    assert np.array_equal(frames[sensor_name]["rgb"], new_rgb)
    assert np.array_equal(frames[sensor_name]["point_cloud"], point_cloud)
    assert frames[sensor_name]["index"] == 2

if __name__ == "__main__":
    pytest.main(['-s', '-v', __file__])
