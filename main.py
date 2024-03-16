import torch.multiprocessing as mp

from environment import Environment
from graphic import Graphic

EPSILON_START = 1.0

def main():

    ## Shared memory

    # Training state
    epsilon = mp.Value('d', EPSILON_START)
    epoch = mp.Value('i', 0)
    loss = mp.Value('d', 0.0)

    # Car state
    speed = mp.Value('i', 0)
    car_action = mp.Value('i', 0)

    # Actions
    cancel_training = mp.Value('b', False)
    save_model = mp.Value('b', False)
    # test model
    # load model
    # pause_rendering

    ## Processes

    p_env_train = mp.Process(target = Environment, args = (epsilon, epoch, loss, speed, car_action, cancel_training, save_model))
    p_graphic = mp.Process()

