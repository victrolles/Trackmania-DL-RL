import torch.multiprocessing as mp

from environment import start_env
from graphic import Graphic

from config import Config

def main():

    config = Config()
    # data bus
    databus_buffer = mp.Queue(maxsize=config.exp_buffer_config.buffer_size)

    # controllers
    end_processes = mp.Value('b', False)
    tm_speed = mp.Value('f', 1.0)
    is_tm_speed_changed = mp.Value('b', False)
    is_training = mp.Value('b', True)
    is_model_saved = mp.Value('b', False)
    is_map_render = mp.Value('b', True)
    is_curves_render = mp.Value('b', True)
    is_random_spawn = mp.Value('b', True)

    ## Processes

    p_env_train = mp.Process(target = start_env, args = (databus_buffer,
                                                         end_processes,
                                                         tm_speed,
                                                         is_training,
                                                         is_model_saved,
                                                         is_map_render,
                                                         is_curves_render,
                                                         is_tm_speed_changed,
                                                         is_random_spawn))
    p_graphic = mp.Process(target = Graphic, args=(databus_buffer,
                                                   end_processes,
                                                   tm_speed,
                                                   is_training,
                                                   is_model_saved,
                                                   is_map_render,
                                                   is_curves_render,
                                                   is_tm_speed_changed,
                                                   is_random_spawn))
    p_env_train.start()
    p_graphic.start()

    p_env_train.join()
    p_graphic.join()

if __name__ == '__main__':
    main()