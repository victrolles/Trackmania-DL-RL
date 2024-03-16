import torch.multiprocessing as mp

def my_function(name):
    print(f'Hello {name} from a process!')

if __name__ == '__main__':
    # Create a Process object and pass the function name and arguments to it
    print("Main process")
    p = mp.Process(target=my_function, args=('World',))
    p.start()  # Starts the process
    p.join()   # Waits for the process to finish