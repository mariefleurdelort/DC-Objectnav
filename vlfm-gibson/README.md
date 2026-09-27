
Project adapted from: 

```
@inproceedings{yokoyama2024vlfm,
  title={VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation},
  author={Naoki Yokoyama and Sehoon Ha and Dhruv Batra and Jiuguang Wang and Bernadette Bucher},
  booktitle={International Conference on Robotics and Automation (ICRA)},
  year={2024}
}
```

## Docker: 

Updated the docker file from original repo and added the DCON docker habitat components to stay consistent with our habitat version (See Docker)
Build the docker image  

Run docker:  

```yaml
  docker run --gpus all -it \
  -v $(pwd)/data:/workspace/vlfm/data \
  -v $(pwd)/vlfm:/workspace/vlfm \
  -v $(pwd)/visualize_trajectory.py:/workspace/vlfm/visualize_trajectory.py \
  -v $(pwd)/eval_with_trajectory.py:/workspace/vlfm/eval_with_trajectory.py \
  vlfm_gibson:latest
```

Run docker commit to keep track of pip changes

## Installations:

Installed vlfm without the extra habitat dependencies \
`cd /workspace/vlfm \
pip install -e .`

--> the pip install caused timm, torch and others to change versions which we DO NOT want, so needed to reinstall the correct versions due to version mismatches with the habitat version we are using
`pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install timm==0.6.7 opencv-python-headless==4.10.0.84
pip install "numpy<2.0"` 

--> updated pyproject.toml to be more flexible with versions (added >) (See pyproject.toml)


## Updated files:

1. Needed update to make the config registration compatible with our habitat version:
/workspace/vlfm/vlfm/policy/habitat_policy.py --> updated to: \
   ```yaml
   @dataclass
    class VLFMPolicyConfig(VLFMConfig, PolicyConfig):
        name: str = "HabitatITMPolicy"
    cs = ConfigStore.instance()
    cs.store(
        group="habitat_baselines/rl/policy",
        name="vlfm_policy",
        node={"HabitatITMPolicy": VLFMPolicyConfig}, 
    )`

3. /workspace/vlfm/config/experiments/vlfm_objectnav_hm3d.yaml --> added max_climb and max_slope to fix habitat version mismatch and set \
   ```yaml
    habitat:
      environment:
        iterator_options:
          max_scene_repeat_steps: 50000
      task:
        success_reward: 2.5
        slack_reward: -1e-3
      simulator:
        agents:
          main_agent:
            max_climb: 0.2
            max_slope: 45.0
    habitat_baselines:
      evaluate: True
      num_environments: 1
      trainer_name: "ver"
      load_resume_state_config: False
      eval:
        should_load_ckpt: False

4. /workspace/vlfm/data/scene_datasets/hm3d --> Created symlink so the simulator \
    `rm -rf /workspace/vlfm/data/scene_datasets/hm3d` \
    `ln -s /workspace/vlfm/data/versioned_data/hm3d-0.2/hm3d /workspace/vlfm/data/scene_datasets/hm3d`

5. IN frontier_exploration package: /opt/conda/envs/DCON/lib/python3.9/site-packages/frontier_exploration/base_explorer.py --> newer habitat-sim requires magnum.Vector3
    - Added import magnum as mn
    - snap_point([realworld_y, ...]) → snap_point(mn.Vector3(realworld_y, ...))
    - snap_point([y, ...]) → snap_point(mn.Vector3(y, ...))

6. IN frontier_exploration package: /opt/conda/envs/DCON/lib/python3.9/site-packages/frontier_exploration/measurements.py --> habitat-sim API changes in newer versions
    - aabb.sizes → aabb.size()
    - aabb.center → aabb.center()
    - center[1] → center.y

7. Updated vlfm_trainer for habitat API version mismatches

8. Updated run.py to make the path work for gibson data


## Data folder: 

Followed the instructions from [the original repository](https://github.com/rai-opensource/vlfm/tree/main) to download all weights
```yaml
wget [https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt](https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt) -O /workspace/vlfm/data/mobile_sam.pt
wget [https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth) -O /workspace/vlfm/data/groundingdino_swint_ogc.pth
wget [https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt) -O /workspace/vlfm/yolov7/yolov7-e6e.pt
wget [https://github.com/rai-opensource/vlfm/raw/main/data/pointnav_weights.pth](https://github.com/rai-opensource/vlfm/raw/main/data/pointnav_weights.pth) -O /workspace/vlfm/data/pointnav_weights.pth
wget [https://github.com/rai-opensource/vlfm/raw/main/data/spot_pointnav_weights.pth](https://github.com/rai-opensource/vlfm/raw/main/data/spot_pointnav_weights.pth) -O /workspace/vlfm/data/spot_pointnav_weights.pth
```
  
Downloaded the 3DSceneGraph_tiny.zip file from: [Redvis 3DSceneGraph](https://sdss.redivis.com/datasets/1kf9-cfjvtqc7q/files) to get the .glb and .navmesh
In the evaluation we are mostly running on the scenes: Collierville, Corozal, Darden, Markleeville, Wiconisco because they have semantic annotations for each scene  

A scene like Cantwell did not work since there are missing npz files for that scene from the 3DSceneGraph_tiny.zip dataset    

All .glb and .navmesh scenes are placed in data/scene_datasets/gibson_semantic  

Unlike HM3D which required a symlink, Gibson scene files were placed directly at data/scene_datasets/gibson_semantic/ alongside a gibson.scene_dataset_config.json file that habitat-sim uses to index the scenes (See data/scene_datasets/gibson_semantic/gibson.scene_dataset_config.yaml)  

data/
scene_datasets/gibson_semantic/     # GLB scene files
  	datasets/objectnav/gibson/       # custom episode JSONs generated

**FIX**: 3DSceneGraph stores object positions as [x, z_hab, y_hab] but habitat-sim expects [x, y_up, z]. This caused all goal positions to be incorrect (swapped Y and Z axes), meaning the agent was always navigating to the wrong location. The fix was to swap the second and third coordinates when reading positions from the NPZ files  

Each json file contains: 
- episode_id
- scene_id: path to the .glb file
- start_position and start_rotation: agent spawn point from the navmesh
- object_category: one of the 6 COCO ObjectNav categories (chair, bed, potted plant, toilet, tv monitor, sofa)
- goals: list of goal objects with their 3D positions and view points

## Config:

- Created a gibson .yaml file in config/benchmark/nav/objectnav/objectnav_gibson.yaml to configure success distance, max steps, target categories (See objectnav_gibson.yaml)
- Created a similar vlfm_objectnac_gibson.yaml file to the one from hm3d in config/experiments/ to configure the full experiments. Main differences: scene dataset path, episode dataset path (See vlfm_objectnav_gibson.yaml)
- Created a gibson.yaml file in config/habitat/dataset/objectnav which finds the episode dataset files for Gibson (see gibson.yaml)

### vlfm/policy:

Updated base_objectnav_policy to set self._visualize=True on every policy instance after construction for the trajectory logger needed for visualization (See base_objectnav_policy)

Running the model with the visualization inside container:  

--> Before running, you need to run: ./scripts/launch_vlm_servers.sh which will start a tmux session and load all the necessary models  

--> Run eval with trajectory + map logging, can change the episode count to any integer  

`python eval_with_trajectory.py habitat_baselines.test_episode_count=5`

--> trajectory files are saved in /workspace/vlfm/data/trajectories

--> Generate visualization PNGs  

`python visualize_trajectory.py data/trajectories/`

--> Copy results out of container on the remote SSH key  

`docker cp <container_id>:/workspace/vlfm/data/trajectories/ /tmp/trajectories/`

--> On your local machine (change marie to your name):  

`scp [name]:/tmp/episode_192.png ~/Desktop/`


## ISSUES:

Multiple Hydra errors occurred due to:  
- Policy config registration (updated habitat_policy.py see page 1)
- max_climb/max_slope error: unexpected key error because of habitat-sim version mismatch. Had to add these to the yaml file
- load _resume_state_config/should_load_ckpt error: tried to load a checkpoint that did not exist so set both variables to False




