# Drones dataset

This repository contains all the code and tools needed to build the "Syndrone" generated using the [CARLA](https://carla.org/) simulator.

The repository is organized into four branches:
1. main: dataset generation code
2. analyses: code used to compute the benchmark semantic segmentation numerical results
3. synth2real: code used to compute the benchmark domain adaptation numerical results
4. detection: code used to compute the benchmark object detection numerical results

You can either download the full dataset [here](https://lttm.dei.unipd.it/paper_data/syndrone/syndrone.zip) or download each sensor from the table below.
- Color ZIPs contain RGB images, Semantic Segmentation labels, Camera Extrinsics, and Bounding Box ground truth.
- Depth ZIPs contain the depth frames
- LiDAR ZIPs contain the LiDAR frames
- [Split ZIP](https://lttm.dei.unipd.it/paper_data/syndrone/splits.zip) contains the lists of samples to use for training and test sets.

|   Town   | Color | Depth | LiDAR |
| :------: | :---: | :---: | :---: |
|  Town01  | [link](https://lttm.dei.unipd.it/paper_data/syndrone/Town01_Opt_120_color.zip) | [link](https://lttm.dei.unipd.it/paper_data/syndrone/Town01_Opt_120_depth.zip) | [link](https://lttm.dei.unipd.it/paper_data/syndrone/Town01_Opt_120_lidar.zip) |
|  Town02  | [link](https://lttm.dei.unipd.it/paper_data/syndrone/Town02_Opt_120_color.zip) | [link](https://lttm.dei.unipd.it/paper_data/syndrone/Town02_Opt_120_depth.zip) | [link](https://lttm.dei.unipd.it/paper_data/syndrone/Town02_Opt_120_lidar.zip) |
|  Town03  | [link](https://lttm.dei.unipd.it/paper_data/syndrone/Town03_Opt_120_color.zip) | [link](https://lttm.dei.unipd.it/paper_data/syndrone/Town03_Opt_120_depth.zip) | [link](https://lttm.dei.unipd.it/paper_data/syndrone/Town03_Opt_120_lidar.zip) |
|  Town04  | [link](https://lttm.dei.unipd.it/paper_data/syndrone/Town04_Opt_120_color.zip) | [link](https://lttm.dei.unipd.it/paper_data/syndrone/Town04_Opt_120_depth.zip) | [link](https://lttm.dei.unipd.it/paper_data/syndrone/Town04_Opt_120_lidar.zip) |
|  Town05  | [link](https://lttm.dei.unipd.it/paper_data/syndrone/Town05_Opt_120_color.zip) | [link](https://lttm.dei.unipd.it/paper_data/syndrone/Town05_Opt_120_depth.zip) | [link](https://lttm.dei.unipd.it/paper_data/syndrone/Town05_Opt_120_lidar.zip) |
|  Town06  | [link](https://lttm.dei.unipd.it/paper_data/syndrone/Town06_Opt_120_color.zip) | [link](https://lttm.dei.unipd.it/paper_data/syndrone/Town06_Opt_120_depth.zip) | [link](https://lttm.dei.unipd.it/paper_data/syndrone/Town06_Opt_120_lidar.zip) |
|  Town07  | [link](https://lttm.dei.unipd.it/paper_data/syndrone/Town07_Opt_120_color.zip) | [link](https://lttm.dei.unipd.it/paper_data/syndrone/Town07_Opt_120_depth.zip) | [link](https://lttm.dei.unipd.it/paper_data/syndrone/Town07_Opt_120_lidar.zip) |
| Town10HD | [link](https://lttm.dei.unipd.it/paper_data/syndrone/Town10HD_Opt_120_color.zip) | [link](https://lttm.dei.unipd.it/paper_data/syndrone/Town10HD_Opt_120_depth.zip) | [link](https://lttm.dei.unipd.it/paper_data/syndrone/Town10HD_Opt_120_lidar.zip) |

## How to run the code

### Requirements

1. Install the [CARLA](https://carla.org/) simulator as modified by [SELMA](https://scanlab.dei.unipd.it/selma-dataset/)
2. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) (or anaconda) and create a new environment with the following command:

    ```bash
    conda env create --name syndrone --file "<project_folder>/extra/carla_env.yml"
    ```

3. Activate the environment with the following command:

    ```bash
    conda activate syndrone
    ```

4. Install the **CARLA python API** provided in the carla zip file with the following command (if you use Windows make sure to change the name to install the appropriate wheel file):

    ```bash
    pip install "<CARLA_installation_folder>/PythonAPI/carla/dist/carla-0.9.12-cp39-cp39-linux_x86_64.whl"
    ```

### Run the code

1. Activate the environment with the following command (if not already activated)):

    ```bash
    conda activate syndrone
    ```

2. Run the CARLA simulator with the following command:

    ```bash
    cd <CARLA_installation_folder>
    ./CarlaUE4.sh
    ```

3. Log the trajectories for each town (for the setup follow the prompt proposed by the code):

    ```bash
    python <project_folder>/log_spectator.py
    ```

4. Once all the required trajectories are logged, run the following command to generate a representative video of each trajectory:

    ```bash
    python <project_folder>/validate_trajectories.py
    ```

5. Generate the dataset with the following command:

    ```bash
    python <project_folder>/run_simulation.py --slen 120 --lidar True --fps 25
    ```

    Arguments:

    - `--pov`: List containing height and angle of the camera, format: [(H1,A1),(H2,A2),...];
    - `--slen`: Length of the acquired sequence in seconds;
    - `--lidar`: Wether to enable logging for the lidar, only heights in [50, 100)m will be used;
    - `--fps`: Fps of the generated data;
    - `--dry`: Wether to start in dry mode, no walkers or vehicles will be spawned.

6. If needed it is possible to render a single sample of the dataset specifying the town and the weather-daytime pair with the following command:

    ```bash
    python <project_folder>/save_frames.py --town <town_number> --weather <weather_daytime_pair> --slen 120 --lidar True --fps 25
    ```

    Arguments:
    - `--town`: Number of the Town world to be loaded;
    - `--weather`: Weather and Daytime to be used for the simulation.
    - `--pov`: List containing height and angle of the camera, format: [(H1,A1),(H2,A2),...];
    - `--slen`: Length of the acquired sequence in seconds;
    - `--lidar`: Wether to enable logging for the lidar, only heights in [50, 100)m will be used;
    - `--fps`: Fps of the generated data;
    - `--dry`: Wether to start in dry mode, no walkers or vehicles will be spawned.

7. Optionally it is possible to generate the videos for each rendered sample with the following command:

    ```bash
    python <project_folder>/parse_to_videos.py --fps 25
    ```

    Arguments:

    - `--fps`: Fps of the generated data;

---

## Credits

This project was created by:

- [Francesco Barbato](https://github.com/barbafrank)
- [Matteo Caligiuri](https://github.com/matteocali)
- [Giulia Rizzoli](https://github.com/rizzoligiulia)

(Dipartimento di Ingegneria dell'Informazione (DEI) - UniPD)
