import torch.multiprocessing as mp

from config.globals import BUFFER_SIZE
from environment import start_env
from graphic import Graphic

def main():

    ## Shared memory

    # Training state
    # epsilon = mp.Value('d', EPSILON_START)
    # epoch = mp.Value('i', 0)
    # loss = mp.Value('d', 0.0)
    # best_dist = mp.Value('d', 0.0)
    # step = mp.Value('i', 0)
    # reward = mp.Value('i', 0)
    # training_time = mp.Value('i', 0)

    # # Car state
    # speed = mp.Value('i', 0)
    # car_action = mp.Value('i', 0)
    # game_time = mp.Value('i', 0)
    # current_dist = mp.Value('d', 0.0)

    # # Actions
    # is_training_mode = mp.Value('b', True)
    # is_model_saved = mp.Value('b', False)
    # game_speed = mp.Value('d', 1.0)
    end_processes = mp.Value('b', False)
    databus_buffer = mp.Queue(maxsize=BUFFER_SIZE)
    # test model
    # load model
    # pause_rendering

    ## Processes

    p_env_train = mp.Process(target = start_env, args = (databus_buffer, end_processes))
    p_graphic = mp.Process(target = Graphic, args=(databus_buffer, end_processes))
    p_env_train.start()
    p_graphic.start()

    p_env_train.join()
    p_graphic.join()

if __name__ == '__main__':
    main()