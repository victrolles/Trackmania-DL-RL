# TrackmaniaAI - Deep Reinforcement Learning

## Setup the environment

### Download TrackMania Nations Forever

You need to get a steam account to download steam app.

Then, you can download Trackmania Nations Forever using the link below :
https://store.steampowered.com/app/11020/TrackMania_Nations_Forever/

### Download Trackmania API

Thanks to Donadigo, a Trackmania API named "TMInterface" is available to connect Trackmania game and python code. You can find the link below :
https://donadigo.com/files/TMInterface/TMInterface_1.4.3_Setup.exe

More informations are available on his repo : https://github.com/donadigo/TMInterfaceClientPython

### Clone the repo

First clone the repo using the command below :
```bash
git clone https://github.com/victrolles/Trackmania-DL-RL.git
```

To run the code, you need **python 3.12.** environment. Download it on your own. \
To make clean project, we will use a virtual environment.

If you want to use **GPUs**, you first need to follow theses steps :
- Install the **CUDA Toolkit** with this link :https://developer.nvidia.com/cuda-toolkit
- Install the **cuDNN** with this link : https://developer.nvidia.com/cudnn
- Start your env and then download pytorch using this command : 
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    ```

In **another** terminal, use the following commands to start the backend :
- Go to the project folder
- Create a virtual environment
    ```bash
    python -m venv .venv
    ```
- Activate the virtual environment
    ```bash
    .venv\Scripts\activate
    ```
- Install the dependencies
    ```bash
    pip install -r requirements.txt
    ```

### Load track on Trackmania

You need to load a track on Trackmania before running the code. You can use the track "snake_map_training" available in the repo.

Copy the file "snake_map_training.Challenge.Gbx" in the folder "C:\Users\%USERNAME%\Documents\TrackMania\Tracks\Challenges\Downloaded". In the repo, the file is located in the address : "extras\maps\snake_map_training\snake_map_training.Challenge.Gbx".

### Load checkpoints on TMInterface

To enable training with random spawn points, you need to load checkpoints on TMInterface. You need to drag and drop all the checkpoint_X.bin files in the folder "C:\Users\%USERNAME%\Documents\TMInterface\States". In the repo, the files are located in the address : "extras\maps\snake_map_training\States".

## Trainer

### TM Inputs
Code in RadarAgent.py : https://github.com/victrolles/Trackmania-DL-RL/blob/DQN_RadarAgent/tm_agents/radar_agent.py

![drawing](https://github.com/user-attachments/assets/a7da6fdc-91cd-4bb9-b561-64f773454ddd)

### Reinforcement Learning Algorithm
Code in DQN : https://github.com/victrolles/Trackmania-DL-RL/tree/DQN_RadarAgent/rl_algorithms/dqn
* Deep Q Network
* LR : 1e-4
* Epsilon Decay : 0.0005
![NN](https://github.com/user-attachments/assets/db88853e-b23f-44d0-a763-d40c14fd8278)

## Results

### First Good Result
![TrackMania Nations Forever (TMInterface 1 4 3) 2024-08-02 21-54-17 (online-video-cutter com)](https://github.com/user-attachments/assets/f89c83af-c25a-48e9-8961-c065848f946a)

![image](https://github.com/user-attachments/assets/96c413f0-0bff-4cef-91f2-e0bee18a8926)

### First Result
![first little results](https://github.com/user-attachments/assets/e5acfffa-65ec-47e3-be9a-0afd1422c729)

### Strange NN degeneration at the end
![first test](https://github.com/user-attachments/assets/e5dbd05d-4ab3-4d33-9de9-5e22fe73196a)

## References
### API
**TrackMania NF API :** https://github.com/donadigo/TMInterfaceClientPython
### Inspirations
* https://www.youtube.com/@yoshtm
* https://www.youtube.com/@linesight-rl
### Assists
*  Book : Deep Reinforcement Learning Hands-on
*  IA : ChatGPT
