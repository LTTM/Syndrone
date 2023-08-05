import os
import cv2 as cv
from tqdm import tqdm
from pathlib import Path
from modules import utils, helpers
from modules.carla_utils import labelmap


if __name__ == "__main__":
    # Load the args
    args = utils.ArgParser(script="parse_to_videos").parse()
    fps = args.fps
    root_dir = args.root_dir

    # Define the colormap to be used
    cmap = labelmap.cmap[...,::-1] #flip to bgr for opencv

    # Load all the dataset folders
    folders = helpers.folders_loader(root_dir)

    # For each main folder (town + slen)
    pbar = tqdm(folders.keys(), desc="Generating the videos")
    for key in pbar:
        pbar.set_description(f"Generating the videos for {key}")

        # Iter over all the weathers
        in_pbar = tqdm(folders[key], desc="Considering the weather-daytime", leave=False)
        for weather_folder in in_pbar:
            in_pbar.set_description(f"Considering the weather-daytime: {weather_folder.stem}")

            # Define the input paths
            rgb_folder = Path(weather_folder / "rgb")
            semantic_folder = Path(weather_folder / "semantic")

            # Define the output paths
            rgb_out_path, semantic_out_path = helpers.videos_out_path(weather_folder)

            # Initialize the videos
            rgb_video = cv.VideoWriter(str(rgb_out_path), cv.VideoWriter.fourcc(*'mp4v'), fps, (1920,1080))
            semantic_video = cv.VideoWriter(str(semantic_out_path), cv.VideoWriter.fourcc(*'mp4v'), fps, (1920,1080))

            # Generate the videos
            fnames = [os.path.splitext(f)[0] for f in sorted(os.listdir(rgb_folder)) if not f.startswith('.') and os.path.isfile(str(rgb_folder / f))]
            for fname in tqdm(fnames, desc="Generating the videos", leave=False):
                rgb_video.write(
                    cv.imread(str(rgb_folder / (fname + ".jpg")), cv.IMREAD_UNCHANGED)
                    )
                semantic_video.write(
                    cmap[
                        cv.imread(str(semantic_folder / (fname + ".png")), cv.IMREAD_UNCHANGED)
                        ]
                    )

            rgb_video.release()
            semantic_video.release()
