"""
test_redis_interface_and_client.py
"""
# %%
import pytest
import numpy as np

from open_perception.communication.redis_interface import RedisInterface
from open_perception.communication.redis_client import RedisClient
from open_perception.utils.config_loader import load_config
import os
import time


# load redis config from default config file
config = load_config()
redis_config = config["communication"]["redis"]

interface = RedisInterface(config=redis_config)
client = RedisClient(config=redis_config)
#%%


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    print("Starting Redis interface and client...")
    interface.start()
    client.connect()
    redis_delay()
    yield
    print("Stopping Redis interface and client...")
    interface.close()
    client.disconnect()

def redis_delay():
    # delay test execution to allow redis to process messages
    time.sleep(0.25)


def test_reset():
    updates = interface.get_updates()
    assert "reset" not in updates
    print(updates)
    client.reset()
    redis_delay()

    updates = interface.get_updates()
    print(updates)
    assert "reset" in updates

    redis_delay()
    updates = interface.get_updates()
    print(updates)
    assert "reset" not in updates

def test_locate_signal():
    obj_name = "object_1"
    img = np.zeros((480, 640, 3), dtype=np.uint8)  # Dummy image

    # test if object names are requested to be located
    elements = client.locate(obj_name, img, wait=False)
    redis_delay()
    objs_to_add = interface.get_classes_to_add()
    objs_to_add_names = [obj["class_name"] for obj in objs_to_add]
    assert obj_name in objs_to_add_names

def test_remove_signal():
    obj_name = "object_1"
    # test if object names are requested to be removed
    client.remove(obj_name)
    redis_delay()
    objs_to_remove = interface.get_classes_to_remove()
    print(objs_to_remove)
    assert obj_name in objs_to_remove


def test_update_config():
    new_config = {"logging": {"level": "DEBUG", "format": "%(asctime)s - %(message)s"}}
    client.update_config(new_config)
    redis_delay()
    updates = interface.get_updates()
    assert "config" in updates

def test_frames():
    sensor_name = "front"
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    meta = {"stamp": time.time(), "id": 1}

    # send rgb frame and check if it is received
    client.send_frame(sensor_name, rgb=rgb, meta=meta, wait=True, fake_pc=False)
    redis_delay()

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
    redis_delay()
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
    redis_delay()
    assert np.array_equal(frames[sensor_name]["rgb"], rgb)
    assert np.array_equal(frames[sensor_name]["point_cloud"], point_cloud)
    assert frames[sensor_name]["index"] == 1
    
    # check if the new synced frames are received
    client.send_frame(sensor_name, point_cloud=point_cloud, meta=meta, wait=True, fake_pc=False)
    redis_delay()
    frames = interface.get_synced_frame_and_pc()
    assert np.array_equal(frames[sensor_name]["rgb"], new_rgb)
    assert np.array_equal(frames[sensor_name]["point_cloud"], point_cloud)
    assert frames[sensor_name]["index"] == 2


if __name__ == "__main__":
    pytest.main(['-s', '-v', __file__])
