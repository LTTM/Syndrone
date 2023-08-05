import carla
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from .. import utils


def out_folders_gen(
    town_name: str,
    weather: tuple[str, carla.WeatherParameters],
    slen: int,
    pov: list,
    use_lidar: bool,
    master_pbar: tqdm = None,
    verbose: bool = False,
    root_dir = "data"
) -> None:
    """
    Generates the output folders for the rendering
    Args:
        - town_name (str): name of the town
        - weather (tuple[str, carla.WeatherParameters]): tuple containing the name and the corresponding weather parameters
        - slen (int): length of the scenario
        - pov (list): list of tuples containing the height and the pitch of the camera
        - master_pbar (tqdm): master progress bar
        - verbose (bool): verbosity flag
    Exceptions:
        - ValueError: if the keyframes file does not exist
    """

    # Define the keyframes folder
    trajectories_folder = Path(root_dir / "trajectories")
    keyframes_file = Path(trajectories_folder / f"keyframes_{town_name}_{slen}.csv")

    if not keyframes_file.exists():
        raise ValueError(
            "The selected keyframes file does not exist (path: %s)" % keyframes_file
        )

    # Define the output folders
    renders_folder = Path(root_dir / "renders")
    out_folder = Path(renders_folder / f"{town_name}_{slen}")
    out_folder_weather = Path(out_folder / weather[0])

    if master_pbar is not None:
        master_pbar.set_description(
            f"Rendering the {weather[0]} scenario, Initializing Folders"
        )

    if out_folder_weather.exists():
        if verbose:
            print("The selected output folder already exists")
            print("Do you want to overwrite it? [y/n]")
            while True:
                choice = input()
                if choice == "y":
                    loader = utils.Loader(
                        "Deleting old data...", "Done!", 0.05, verbose
                    ).start()
                    shutil.rmtree(out_folder_weather)
                    loader.stop()
                    break
                elif choice == "n":
                    print("Exiting...")
                    exit()
                else:
                    print("You have inserted a wrong option, retry.")
        else:
            shutil.rmtree(out_folder_weather)

    bboxes_folder = Path(out_folder_weather / "bboxes")
    bboxes_folder.mkdir(parents=True, exist_ok=True)

    rgb_folders = {}
    depth_folders = {}
    semantic_folders = {}
    lidar_folders = {}
    camera_folders = {}
    for h, _ in pov:
        rgb_folder = Path(out_folder_weather / ("height%dm" % h) / "rgb")
        rgb_folder.mkdir(parents=True, exist_ok=True)
        rgb_folders[h] = rgb_folder

        depth_folder = Path(out_folder_weather / ("height%dm" % h) / "depth")
        depth_folder.mkdir(parents=True, exist_ok=True)
        depth_folders[h] = depth_folder

        semantic_folder = Path(out_folder_weather / ("height%dm" % h) / "semantic")
        semantic_folder.mkdir(parents=True, exist_ok=True)
        semantic_folders[h] = semantic_folder

        camera_folder = Path(out_folder_weather / ("height%dm" % h) / "camera")
        camera_folder.mkdir(parents=True, exist_ok=True)
        camera_folders[h] = camera_folder

        if use_lidar and 50 < h < 100:
            lidar_folder = Path(out_folder_weather / ("height%dm" % h) / "lidar")
            lidar_folder.mkdir(parents=True, exist_ok=True)
            lidar_folders[h] = lidar_folder

    return (
        keyframes_file,
        rgb_folders,
        depth_folders,
        semantic_folders,
        lidar_folders,
        camera_folders,
        bboxes_folder,
    )


def keyframes_loader(keyframes_file: Path, verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads the keyframes from the keyframes file
    Args:
        - keyframes_file (Path): path to the keyframes file
        - verbose (bool): verbosity flag
    Return:
        - kframes (np.ndarray): array containing the keyframes
        - skframes (np.ndarray): array containing the keyframes
    """

    kframes = np.array(
        [[float(f.strip()) for f in l.strip().split(",")] for l in open(keyframes_file)]
    )
    if verbose:
        print("Loaded Keyframes, shape:", kframes.shape)
    skframes = np.zeros((kframes.shape[0], 5))
    skframes[:, :2] = kframes[:, :2]  # x, y
    skframes[:, 2] = kframes[:, 6]  # z0: floor height
    skframes[:, 3] = np.cos(np.pi * kframes[:, 4] / 180)
    skframes[:, 4] = np.sin(np.pi * kframes[:, 4] / 180)

    return kframes, skframes


def folders_loader(root_dir = "data") -> dict[str, list[Path]]:
    """
    Function that load all the folder path of the dataset
    Return:
        - folders_dict (dict{str, list[Path]}): dictionary containing as keys the name of the town and as values the list of the weather folders
    """

    # Initialize the dictionary
    folders_dict = {}
    
    # Define the path of the dataset
    dataset_folder = Path(root_dir / "renders")

    # Define the list of the towns_folders
    towns_folder = [folder for folder in dataset_folder.iterdir() if folder.is_dir()]
    
    # For each town_folder
    for town_folder in towns_folder:
        # Define the list of the weather_folders
        weather_folders = [folder / "height50m" for folder in town_folder.iterdir() if folder.is_dir()]

        # Add the town_folder to the dictionary
        folders_dict[str(town_folder.stem)] = weather_folders

    return folders_dict


def videos_out_path(folder: Path) -> tuple[Path, Path]:
    """
    Generates the output folders for the videos
    Args:
        - folder (Path): path to the main folder containing the renders
    """

    out_folder = Path(folder / "videos")
    out_folder.mkdir(parents=True, exist_ok=True)

    rgb_path = Path(out_folder / "rgb.mp4")
    semantic_path = Path(out_folder / "semantic.mp4")


    return rgb_path, semantic_path
