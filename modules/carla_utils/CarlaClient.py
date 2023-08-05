import carla
import numpy as np
import time
import json
import cv2 as cv
from scipy import interpolate
from tqdm import tqdm
from pathlib import Path
from .spawn_utils import spawn_walkers, spawn_vehicles
from .custom_bp_tags import override_parked_vehicles, spawn_static_vehicles, get_corners_from_bb
from .carla_sync import CarlaSyncMode
from .labelmap import bblabels
from .binary_ply import carla_lidarbuffer_to_ply
from .. import utils
from .. import helpers


def verbose_parse(s: str) -> tuple[bool, bool]:
    """
    Parses the verbose argument
    Args:
        - s (str): the string to parse
    Returns:
        - verbose (bool): wether to show the output
        - opt_pbar (bool): wether to show the progress bar
    """
    
    if s == "off":
        verbose = False
        opt_pbar = False
    elif s == "minimal":
        verbose = False
        opt_pbar = True
    elif s == "all":
        verbose = True
        opt_pbar = True
    else:
        raise ValueError("Invalid verbose argument")

    return verbose, opt_pbar


class CarlaClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 2000, timeout: float = 20):
        """
        Initializes the CarlaClient object
        Args:
            - host (str): IP address of the CARLA server
            - port (int): port of the CARLA server
            - timeout (float): timeout for the connection to the CARLA server
        """
        self.client = carla.Client(host=host, port=port)
        self.client.set_timeout(timeout)

    def connect(self) -> None:
        """
        Connects to the CARLA server
        """

        try:
            self.client.get_server_version()
        except RuntimeError as e:
            raise e

    def load_world(self, name: str, weather: carla.WeatherParameters) -> carla.World:
        """
        Loads the specified world
        Args:
            - world_name (str): name of the world to load
            - weather (carla.WeatherParameters): weather parameters to set
        Returns:
            - world (carla.World): the loaded world
        """

        self.world = self.client.load_world(name)
        self.world.set_weather(weather)

        return self.world

    def get_client(self) -> carla.Client:
        """
        Gets the client
        Returns:
            - client (carla.Client): the client
        """

        return self.client

    def get_traffic_manager(self, port: int = 8000) -> carla.TrafficManager:
        """
        Gets the traffic manager
        Args:
            - port (int): port of the traffic manager
        Returns:
            - tm (carla.TrafficManager): the traffic manager
        """

        tm = self.client.get_trafficmanager(port)

        return tm

    def render_scene(
        self,
        folders: list,
        town_name: str,
        pov: list,
        weather: tuple,
        slen: int,
        fps: int,
        use_lidar: bool,
        dry: bool,
        verbose: str = "of",
        master_pbar: tqdm = None
    ) -> None:
        """
        Render the scene
        Args:
            - folders (list): list of the output folders
            - town_name (str): name of the town to be used
            - pov (list): list of tuples containing the height and angle of the camera
            - weather (tuple): tuple containing the name and the id of the weather to be used
            - slen (int): length of the sequence to be rendered
            - fps (int): fps of the sequence to be rendered
            - use_lidar (bool): wether to use the lidar sensor
            - dry (bool): wether to start in dry mode
            - verbose (str): wether to print the output of the script (of, minimal, all)
            - master_pbar (tqdm): master progress bar
        Exceptions:
            - ValueError: raised if the selected keyframes file does not exist
        """

        # Parse the verbose argument
        verbose, opt_pbar = verbose_parse(verbose)

        # If not verbose, redirect stdout and stderr to /dev/null
        if not verbose:
            print_manager = utils.PrintManager(filter_errors=False)
            print_manager.disable_print()

        # Define and generate the output folders
        (
            keyframes_file,
            rgb_folders,
            depth_folders,
            semantic_folders,
            lidar_folders,
            camera_folders,
            bboxes_folder,
        ) = folders

        # Load the keyframes
        kframes, skframes = helpers.keyframes_loader(keyframes_file)

        # Define the interpolation function
        curve = interpolate.make_interp_spline(
            np.arange(kframes.shape[0]), skframes, k=3
        )

        # Setup CARLA
        print("--------------------------------------------------------------------")
        print("CARLA Setup...")

        if master_pbar is not None:
            master_pbar.set_description(
                f"Rendering the {weather[0]} scenario, Setting up Carla"
            )

        print("\nLoading the world...")
        loader = utils.Loader("Loading...", "Done!", 0.05, verbose).start()
        world = self.load_world(name=town_name, weather=weather[1])
        bp_lib = world.get_blueprint_library()
        loader.stop()

        if master_pbar is not None:
            master_pbar.set_description(
                f"Rendering the {weather[0]} scenario, Spawning Static Vehicles"
            )

        print("\nLoading the vehicles...")
        parked = override_parked_vehicles(world, bp_lib, verbose=verbose)
        static = spawn_static_vehicles(world, world.get_map().name, bp_lib)

        if master_pbar is not None:
            master_pbar.set_description(
                f"Rendering the {weather[0]} scenario, Spawning Walkers"
            )

        # Spawn the walkers
        # must be spawned in asynchronous mode due to carla bug
        if not dry:
            print("\nSpawning the pedestrians...")
            pedestrians, _ = spawn_walkers(world, verbose=verbose)
        else:
            pedestrians, _ = [], []

        if master_pbar is not None:
            master_pbar.set_description(
                f"Rendering the {weather[0]} scenario, Initializing Sensors"
            )

        print("\nLoading the sensors...")
        loader = utils.Loader("Loading...", "Done!", 0.05, verbose).start()
        classes = carla.CityObjectLabel.__dict__  # name: classid
        sbboxes = []
        level_bbs = {
            k: world.get_level_bbs(classes[k]) or None for k in bblabels
        }
        for k, v in level_bbs.items():
            if v is not None:
                for bb in v:
                    sbboxes.append(
                        {
                            "corners": get_corners_from_bb(bb),
                            "class": [bblabels[k]],
                            "id": len(sbboxes),
                        }
                    )

        bp_lib = world.get_blueprint_library()
        bp_rgb = bp_lib.filter("sensor.camera.rgb")[0]
        bp_depth = bp_lib.filter("sensor.camera.depth")[0]
        bp_semantic = bp_lib.filter("sensor.camera.semantic_segmentation")[0]
        bp_lidar = bp_lib.filter("lidar")[0]

        intrinsics = {"image_size_x": 1920, "image_size_y": 1080, "fov": 90}
        postprocess = {
            "enable_postprocess_effects": True,
            "chromatic_aberration_intensity": "0.5",
            "chromatic_aberration_offset": "0",
        }
        lidarparams = {
            "channels": 64,
            "points_per_second": 100000 * fps,
            "rotation_frequency": fps,
            "range": 100,
            "lower_fov": -89,
            "upper_fov": -49
        }

        for bp in [bp_rgb, bp_depth, bp_semantic]:
            for k, v in intrinsics.items():
                bp.set_attribute(k, str(v))
        for k, v in postprocess.items():
            bp_rgb.set_attribute(k, str(v))
        for k, v in lidarparams.items():
            bp_lidar.set_attribute(k, str(v))

        spectator = world.get_spectator()
        sensors = [
            (
                world.spawn_actor(bp_rgb, spectator.get_transform()),
                world.spawn_actor(bp_depth, spectator.get_transform()),
                world.spawn_actor(bp_semantic, spectator.get_transform()),
            )
            + (
                (world.spawn_actor(bp_lidar, spectator.get_transform()),)
                if use_lidar and 50 < h < 100
                else ()
            )
            for (h, _) in pov
        ]
        loader.stop()
        time.sleep(1)

        with CarlaSyncMode(
            world, *[s for ss in sensors for s in ss], fps=fps
        ) as sync:
            if master_pbar is not None:
                master_pbar.set_description(
                    f"Rendering the {weather[0]} scenario, Enabled Sync Mode - Spawning Vehicles"
                )
            if not dry:
                print("\nSpawning the vehicles...")
                tm = self.get_traffic_manager(8000)
                vehicles = spawn_vehicles(world, tm, verbose=verbose)
            else:
                vehicles = []

            print(
                "\n--------------------------------------------------------------------"
            )
            print("Starting the simulation...")

            if verbose:
                pbar = tqdm(
                    np.linspace(0, skframes.shape[0] - 1, fps * slen),
                    desc="Replaying drone trajectory...",
                )
            elif opt_pbar:
                pbar = tqdm(
                        np.linspace(0, skframes.shape[0] - 1, fps * slen),
                        desc="Replaying drone trajectory...", leave=False
                    )
            else:
                pbar = np.linspace(0, skframes.shape[0] - 1, fps * slen)

            if master_pbar is not None:
                master_pbar.set_description(
                    f"Rendering the {weather[0]} scenario, Running Simulation"
                )
                

            for f, t in enumerate(pbar):
                x, y, z0, cyaw, syaw = curve(t)
                yaw = 180 * np.arctan2(syaw, cyaw) / np.pi
                tr = carla.Transform(
                    carla.Location(x, y, z0 + 20), carla.Rotation(-30, yaw, 0)
                )
                spectator.set_transform(tr)

                for ss, (h, a) in zip(sensors, pov):
                    if use_lidar and 50 < h < 100:
                        rgb, depth, semantic, lidar = ss
                        tr = carla.Transform(
                            carla.Location(x, y, z0 + h), carla.Rotation(0, 90+yaw, 0)
                        )
                        lidar.set_transform(tr)
                    else:
                        rgb, depth, semantic = ss

                    tr = carla.Transform(
                        carla.Location(x, y, z0 + h), carla.Rotation(a, yaw, 0)
                    )
                    rgb.set_transform(tr)
                    depth.set_transform(tr)
                    semantic.set_transform(tr)

                data = sync.tick()[1:]
                bboxes = [] + sbboxes
                for actor in vehicles + pedestrians + parked + static:
                    try:
                        bboxes.append(
                            {
                                "corners": get_corners_from_bb(
                                    actor.bounding_box, actor.get_transform()
                                ),
                                "class": actor.semantic_tags,
                                "id": actor.id,
                            }
                        )
                    except RuntimeError:
                        # print('An actor as died')
                        pass
                with open(str(Path(bboxes_folder / ("%05d.json" % f))), "w") as fout:
                    json.dump(bboxes, fout)

                # parse the flattened sensors back in the correct shape
                sdata = []
                idx = 0
                for ss in sensors:
                    sdata.append(data[idx : idx + len(ss)])
                    idx += len(ss)

                for sd, (h, a) in zip(sdata, pov):
                    if use_lidar and 50 < h < 100:
                        im, d, lb, l = sd
                        Npts = 0
                        for ch in range(l.channels):
                            Npts += l.get_point_count(ch)
                        carla_lidarbuffer_to_ply(
                            str(Path(lidar_folders[h] / ("%05d.ply" % f))),
                            Npts,
                            l.raw_data,
                            flip_xy='y'
                        )
                    else:
                        im, d, lb = sd
                    im = np.frombuffer(im.raw_data, dtype=np.uint8).reshape(
                        im.height, im.width, 4
                    )[..., :3]
                    d = np.frombuffer(d.raw_data, dtype=np.uint8).reshape(
                        d.height, d.width, 4
                    )[..., :3]
                    d = (d.astype(int) * [256 * 256, 256, 1]).sum(axis=-1) / (
                        256 * 256 * 256 - 1.0
                    )
                    d = ((256 * 256 - 1) * (d)).astype(np.uint16)
                    lb = np.frombuffer(lb.raw_data, dtype=np.uint8).reshape(
                        lb.height, lb.width, 4
                    )[..., 2]

                    cv.imwrite(str(Path(rgb_folders[h] / ("%05d.jpg" % f))), im)
                    cv.imwrite(str(Path(depth_folders[h] / ("%05d.png" % f))), d)
                    cv.imwrite(str(Path(semantic_folders[h] / ("%05d.png" % f))), lb)

                    with open(Path(camera_folders[h] / ("%05d.json" % f)), "w") as fout:
                        json.dump(
                            {"x": x, "y": y, "z": h, "pitch": a, "yaw": yaw, "roll": 0},
                            fout,
                        )

        print("Done!")
        print("--------------------------------------------------------------------")

        # If not verbose, restore the stdout and stderr
        if not verbose:
            print_manager.enable_print()
