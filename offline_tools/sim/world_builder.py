"""
Generates SDF world files for Gazebo Harmonic physics simulation.
Stitches satellite map tiles into an albedo texture map, applies PBR materials to the ground plane,
and includes a Quadcopter Drone model with downward-facing camera.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import yaml
from PIL import Image
from shared.logging_cfg import setup_logger

logger = setup_logger("world_builder")

SDF_TEMPLATE = """<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="{world_name}">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <!-- Directional Sun Light -->
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 200 0 0 0</pose>
      <diffuse>1.0 1.0 1.0 1</diffuse>
      <specular>0.3 0.3 0.3 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Textured Satellite Ground Plane -->
    <model name="satellite_ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>1000 1000</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>1000 1000</size>
            </plane>
          </geometry>
          <material>
            <ambient>1 1 1 1</ambient>
            <diffuse>1 1 1 1</diffuse>
            <pbr>
              <metal>
                <albedo_map>{texture_path}</albedo_map>
              </metal>
            </pbr>
          </material>
        </visual>
      </link>
    </model>

    <!-- Quadcopter Drone Model with Downward Camera -->
    <model name="x500_quadcopter">
      <pose>0 0 10 0 0 0</pose>
      <link name="base_link">
        <inertial>
          <mass>1.5</mass>
          <inertia>
            <ixx>0.03</ixx><ixy>0</ixy><ixz>0</ixz>
            <iyy>0.03</iyy><iyz>0</iyz>
            <izz>0.06</izz>
          </inertia>
        </inertial>
        <visual name="body_visual">
          <geometry>
            <box>
              <size>0.5 0.5 0.15</size>
            </box>
          </geometry>
          <material>
            <ambient>0.1 0.1 0.8 1</ambient>
            <diffuse>0.2 0.3 0.9 1</diffuse>
          </material>
        </visual>
        <sensor name="downward_camera" type="camera">
          <pose>0 0 -0.1 0 1.5707963 0</pose>
          <camera>
            <horizontal_fov>1.047</horizontal_fov>
            <image>
              <width>640</width>
              <height>480</height>
            </image>
            <clip>
              <near>0.1</near>
              <far>1000</far>
            </clip>
          </camera>
          <always_on>1</always_on>
          <update_rate>30</update_rate>
          <visualize>true</visualize>
        </sensor>
      </link>
    </model>

  </world>
</sdf>
"""


def stitch_tiles_to_texture(tile_dir: str, output_path: str = "data/ajabgarh_sat.png") -> str:
    """Stitches downloaded satellite map tiles into a composite texture map image."""
    abs_out_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_out_path), exist_ok=True)

    if not os.path.exists(tile_dir):
        logger.warning(f"Tile directory {tile_dir} not found. Generating default satellite texture.")
        img = Image.new("RGB", (1024, 1024), color=(40, 120, 40))
        img.save(abs_out_path)
        return abs_out_path

    files = [f for f in os.listdir(tile_dir) if f.endswith(".png")]
    if not files:
        logger.warning("No tile images found in tile directory. Generating default texture.")
        img = Image.new("RGB", (1024, 1024), color=(40, 120, 40))
        img.save(abs_out_path)
        return abs_out_path

    tiles = []
    for f in files:
        parts = f.replace(".png", "").split("_")
        if len(parts) == 3:
            tiles.append((int(parts[1]), int(parts[2]), os.path.join(tile_dir, f)))

    xs = [t[0] for t in tiles]
    ys = [t[1] for t in tiles]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    cols = max_x - min_x + 1
    rows = max_y - min_y + 1
    tile_px = 256

    composite = Image.new("RGB", (cols * tile_px, rows * tile_px))
    for x, y, filepath in tiles:
        try:
            tile_img = Image.open(filepath)
            px_x = (x - min_x) * tile_px
            px_y = (y - min_y) * tile_px
            composite.paste(tile_img, (px_x, px_y))
        except Exception:
            pass

    # Resize composite image to 2048x2048 for optimal GPU texture rendering
    composite_resized = composite.resize((2048, 2048), getattr(Image, 'LANCZOS', getattr(Image, 'Resampling', Image).LANCZOS))
    composite_resized.save(abs_out_path)
    logger.info(f"Stitched {len(tiles)} satellite map tiles into texture {abs_out_path} (2048x2048)")
    return abs_out_path


def generate_sdf_world(config_path: str, output_sdf_path: str = "data/sim_world.sdf"):
    """Generates SDF simulation world file with textured satellite ground plane & drone model."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    sim_cfg = cfg.get("sim", {})
    world_name = sim_cfg.get("world_name", "ajabgarh_world")
    tile_dir = cfg.get("map", {}).get("tile_dir", "data/tiles")
    texture_path = sim_cfg.get("satellite_texture_path", "data/ajabgarh_sat.png")

    abs_texture_path = stitch_tiles_to_texture(tile_dir, texture_path)
    sdf_content = SDF_TEMPLATE.format(world_name=world_name, texture_path=abs_texture_path)

    os.makedirs(os.path.dirname(output_sdf_path), exist_ok=True)
    with open(output_sdf_path, "w") as f:
        f.write(sdf_content)

    logger.info(f"Generated Gazebo SDF world file at {output_sdf_path} with satellite texture.")


if __name__ == "__main__":
    generate_sdf_world("config/sim.yaml")
