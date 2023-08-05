import numpy as np
import sys
import carla
import cv2 as cv
import random
from scipy import interpolate
from tqdm import tqdm
from pathlib import Path
from modules import carla_utils, helpers


# Define the random seed
random.seed(12345)

print("\n=== TRAJECTORIES VALIDATION SCRIPT ===\n")

# Define the keyframes folder
trajectories_folder = Path("data/trajectories")
keyframes_files = list(trajectories_folder.iterdir())
if not trajectories_folder.exists() or len(keyframes_files) == 0:
    print("There are no trajectories to validate")
    sys.exit(1)

# Remove from the list the files which last modified date are lower than the one of the respective video
# Correct it to properly work also on windows
for keyframes_file in keyframes_files:
    video_file = Path("data/trajectories_val_videos") / f"{str(keyframes_file.stem)}.mp4"
    if video_file.exists() and keyframes_file.stat().st_mtime < video_file.stat().st_mtime:
        keyframes_files.remove(keyframes_file)

# Define the output folders
trajectories_val_videos_folder = Path("data/trajectories_val_videos")
trajectories_val_videos_folder.mkdir(parents=True, exist_ok=True)

# Define the height and angle of the POV
h = 50
a = -60

pbar = tqdm(keyframes_files, desc="Rendering trajectories...", leave=False)
for keyframes_file in pbar:
    # Extract the sequence parameters
    town_name = keyframes_file.stem.split("_")[1] + "_Opt"
    dur = int(keyframes_file.stem.split("_")[3])
    fps = 25

    # Create video
    video_rgb = cv.VideoWriter(str(Path(trajectories_val_videos_folder / f"{str(keyframes_file.stem)}.mp4")), cv.VideoWriter.fourcc(*'mp4v'), 25, (1280, 720), True)

    # Update bar description
    pbar.set_description(f"Rendering trajectories for {town_name} with dur {dur}s...")

    # Load the keyframes
    kframes, skframes = helpers.keyframes_loader(keyframes_file, verbose=True)

    # Define the interpolation function
    curve = interpolate.make_interp_spline(np.arange(kframes.shape[0]), skframes, k=3)

    # Setup CARLA
    client = carla_utils.CarlaClient(timeout=3000)
    try:
        client.connect()
    except RuntimeError as e:
        print("Could not connect to CARLA server:\n\t", e)
        sys.exit(1)

    world = client.load_world(name=town_name, weather=carla.WeatherParameters.ClearNoon)
    bp_lib = world.get_blueprint_library()

    classes = carla.CityObjectLabel.__dict__  # name: classid
    bp_lib = world.get_blueprint_library()
    bp_rgb = bp_lib.filter('sensor.camera.rgb')[0]

    intrinsics = {'image_size_x': 1280, 'image_size_y': 720, 'fov': 90}
    postprocess = {'enable_postprocess_effects': True, 
                   'chromatic_aberration_intensity': '0.5',
                   'chromatic_aberration_offset': '0'}
    
    for k, v in intrinsics.items():
        bp_rgb.set_attribute(k, str(v))

    spectator = world.get_spectator()
    sensor = world.try_spawn_actor(bp_rgb, spectator.get_transform())
    
    with carla_utils.CarlaSyncMode(world, sensor, fps=fps, render=True) as sync:
        pbar = tqdm(np.linspace(0, skframes.shape[0]-1, fps*dur), desc="Replaying drone trajectory...", leave=False)
        for f, t in enumerate(pbar):
            x, y, z0, cyaw, syaw = curve(t)
            yaw = 180*np.arctan2(syaw, cyaw)/np.pi
            tr = carla.Transform(carla.Location(x, y, z0+h), carla.Rotation(a, yaw, 0))
            spectator.set_transform(tr)
            sensor.set_transform(tr)

            try:
                data = sync.tick()[1]
            except:
                print("Missed frame")

            im = np.frombuffer(data.raw_data, dtype=np.uint8).reshape(data.height, data.width, 4)[...,:3]
            video_rgb.write(im)
    video_rgb.release()

print("Done!")
print("--------------------------------------------------------------------")
