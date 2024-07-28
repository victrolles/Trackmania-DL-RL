import torch.multiprocessing as mp

from config.globals import BUFFER_SIZE
from environment import start_env
from graphic import Graphic

def main():

    # data bus
    databus_buffer = mp.Queue(maxsize=BUFFER_SIZE)

    # controllers
    end_processes = mp.Value('b', False)
    tm_speed = mp.Value('f', 1.0)
    is_training = mp.Value('b', True)
    save_model = mp.Value('b', False)
    is_map_render = mp.Value('b', True)
    is_curves_render = mp.Value('b', True)

    ## Processes

    p_env_train = mp.Process(target = start_env, args = (databus_buffer, end_processes, tm_speed, is_training, save_model, is_map_render, is_curves_render))
    p_graphic = mp.Process(target = Graphic, args=(databus_buffer, end_processes, tm_speed, is_training, save_model, is_map_render, is_curves_render))
    p_env_train.start()
    p_graphic.start()

    p_env_train.join()
    p_graphic.join()

if __name__ == '__main__':
    main()