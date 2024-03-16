import torch.multiprocessing as mp

from environment import start_env
from graphic import Graphic
from config import EPSILON_START

def main():

    ## Shared memory

    # Training state
    epsilon = mp.Value('d', EPSILON_START)
    epoch = mp.Value('i', 0)
    loss = mp.Value('d', 0.0)
    best_dist = mp.Value('d', 0.0)
    current_dist = mp.Value('d', 0.0)
    buffer_size = mp.Value('i', 0)

    # Car state
    speed = mp.Value('i', 0)
    car_action = mp.Value('i', 0)
    time = mp.Value('i', 0)

    # Actions
    cancel_training = mp.Value('b', False)
    save_model = mp.Value('b', False)
    end_processes = mp.Value('b', False)
    # test model
    # load model
    # pause_rendering

    ## Processes

    p_env_train = mp.Process(target = start_env, args = (epsilon, epoch, loss, best_dist, current_dist, buffer_size, speed, car_action, time, cancel_training, save_model, end_processes))
    p_graphic = mp.Process(target = Graphic, args=(epsilon, epoch, loss, best_dist, current_dist, buffer_size, speed, car_action, time, cancel_training, save_model, end_processes))

    p_env_train.start()
    p_graphic.start()

    p_env_train.join()
    p_graphic.join()

if __name__ == '__main__':
    main()