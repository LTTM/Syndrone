import argparse
import carla
from pathlib import Path


# Constants
TOWNS_NUMBER = ["01", "02", "03", "04", "05", "06", "07", "10"]
OPT_TOWNS_NUMBER = ["1", "2", "3", "4", "5", "6", "7"]
TOWNS_NAMES = ["Town01_Opt", "Town02_Opt", "Town03_Opt", "Town04_Opt", "Town05_Opt", "Town06_Opt", "Town07_Opt", "Town10HD_Opt"]
WEATHERS = ["Default", "ClearNoon", "CloudyNoon", "WetNoon", "WetCloudyNoon", 
            "MidRainyNoon", "HardRainNoon", "SoftRainNoon", "ClearSunset", 
            "CloudySunset", "WetSunset", "WetCloudySunset", "MidRainSunset", 
            "HardRainSunset", "SoftRainSunset"]
DEFAULT_POV = "[(20,-30),(50,-60),(80,-90)]"


def str2town(s: str) -> str:
    """
    Parses a string of the form "X" into the proper town name "TownX"
    Args:
        - s (str): string to be parsed
    Return:
        - town_name (str): town name 
    """

    town_number = "0" + s if s in OPT_TOWNS_NUMBER else s
    town_number = s + "HD" if s == "10" else town_number
    return "Town" + town_number + "_Opt"


def str2tuplelist(s) -> list:
    """
    Parses a string of the form [(X1,Y1),(X2,Y2),...] into a list of tuples
    Args:
        - s (str): string to be parsed
    Return:
        - ts (list): list of tuples
    """

    s = s.lstrip("[").rstrip("]")
    ts = [tuple(float(n) for n in t.lstrip(",").lstrip("(").split(",")) for t in s.split(")") if len(t)>0]
    return ts


def str2bool(s) -> bool:
    """
    Parses a string into a boolean
    Args:
        - s (str): string to be parsed
    Return:
        - bool (bool): boolean
    """

    return s.lower() in ["1", "t", "true", "y", "yes"]


def str2path(s: str) -> Path:
    """
    Parses a string into a Path object
    Args:
        - s (str): string to be parsed
    Return:
        - Path (Path): Path object
    """

    return Path(s)


def check_weather(s: str) -> None:
    """
    Checks if the weather parameters are valid
    Args:
        - s (str): string to be parsed
    """

    if s not in WEATHERS:
        raise argparse.ArgumentTypeError("Weather parameters not valid")


def weatherparameters(s) -> tuple:
    """
    Parses a string into a tuple of the form (s, carla.WeatherParameters.__dict__[s])
    Args:
        - s (str): string to be parsed
    Return:
        - (s, carla.WeatherParameters.__dict__[s]) (tuple): tuple containing the name and the corresponding weather parameters
    """

    try:
        check_weather(s)
    except argparse.ArgumentTypeError as e:
        raise e

    return s, carla.WeatherParameters.__dict__[s]


def load_parser_save_frames() -> argparse.ArgumentParser:
    """
    Creates an ArgumentParser object to parse the arguments for the save_frames.py script
    Return:
        - parser (ArgumentParser): ArgumentParser object
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--town", type=str2town, choices=TOWNS_NAMES, required=True, help="Number of the Town world to be loaded")
    parser.add_argument("-p", "--pov", type=str2tuplelist, default=DEFAULT_POV, 
                        help="List containing height and angle of the camera, format: [(H1,A1),(H2,A2),...]")
    parser.add_argument("-w", "--weather", type=weatherparameters, default="ClearNoon", 
                        help="Weather and Daytime to be used for the simulation")
    parser.add_argument("-s", "--slen", type=int, default=120, help="Length of the acquired sequence in seconds")
    parser.add_argument("-l", "--lidar", type=str2bool, default=False, help="Wether to enable logging for the lidar, only heights < 100m will be used")
    parser.add_argument("-f", "--fps", type=int, default=25, help="fps of the generated data")
    parser.add_argument("-d", "--dry", type=str2bool, default=False, help="Wether to start in dry mode, no walkers or vehicles will be spawned")

    return parser


def load_parser_run_simulations() -> argparse.ArgumentParser:
    """
    Creates an ArgumentParser object to parse the arguments for the run_simulations.py script
    Return:
        - parser (ArgumentParser): ArgumentParser object
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--pov", type=str2tuplelist, default=DEFAULT_POV, 
                        help="List containing height and angle of the camera, format: [(H1,A1),(H2,A2),...]")
    parser.add_argument("-s", "--slen", type=int, default=120, help="Length of the acquired sequence in seconds")
    parser.add_argument("-l", "--lidar", type=str2bool, default=False, help="Wether to enable logging for the lidar, only heights < 100m will be used")
    parser.add_argument("-f", "--fps", type=int, default=25, help="fps of the generated data")
    parser.add_argument("-d", "--dry", type=str2bool, default=False, help="Wether to start in dry mode, no walkers or vehicles will be spawned")

    return parser


def load_parser_parse_to_videos() -> argparse.ArgumentParser:
    """
    Creates an ArgumentParser object to parse the arguments for the parse_to_videos.py script
    Return:
        - parser (ArgumentParser): ArgumentParser object
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--fps", type=int, default=25, help="fps of the generated data")
    parser.add_argument("--root_dir", type=Path, default="data", help="root folder for the dataset")

    return parser


class ArgParser():

    def __init__(self, script: str) -> None:
        if script == "save_frames":
            self.parser = load_parser_save_frames()
        elif script == "run_simulation":
            self.parser = load_parser_run_simulations()
        elif script == "parse_to_videos":
            self.parser = load_parser_parse_to_videos()
        else:
            raise ValueError("The script name is not valid")
        
        self.args = None
    
    def parse(self) -> argparse.Namespace:
        # Parse all the argument
        self.args = self.parser.parse_args()

        return self.args
        
