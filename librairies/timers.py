import time

class Timers:
    
        def __init__(self):
            self.timers = {}
    
        def add_timer(self, name: str):
            self.timers[name] = Timer(name)
    
        def start(self, name: str):
            self.timers[name].start()
    
        def stop(self, name: str):
            self.timers[name].stop()
    
        def pause(self, name: str):
            self.timers[name].pause()
    
        def resume(self, name: str):
            self.timers[name].resume()
    
        def get_time(self, name: str) -> float:
            return self.timers[name].get_time()
    
        def get_string_time(self, name: str) -> str:
            return self.timers[name].get_string_time()
    
        def __str__(self):
            string = ""
            for timer in self.timers.values():
                string += str(timer) + "\n"
            return string

class Timer:

    def __init__(self, name: str):
        self.name = name
        self.start_time = 0
        self.paused_time = 0
        self.is_running = False

    def __str__(self):
        return f"Name: {self.name}, time: {self.get_string_time()}, is_running: {self.is_running}"

    def start(self):
        self.start_time = time.time()
        self.paused_time = 0
        self.is_running = True

    def stop(self):
        self.is_running = False
        self.paused_time = 0

    def pause(self):
        if self.is_running:
            self.paused_time = time.time() - self.start_time
            self.is_running = False

    def resume(self):
        if not self.is_running and self.paused_time > 0:
            self.start_time = time.time() - self.paused_time
            self.is_running = True
            self.paused_time = 0

    def get_time(self) -> float:
        if self.is_running:
            return time.time() - self.start_time
        else:
            return self.paused_time

    def get_string_time(self) -> str:
        total_time = self.get_time()

        hour = int(total_time / 3600)
        minute = int(total_time / 60)
        seconde = int(total_time % 60)

        if hour > 0:
            string = f"{hour}h {minute}min {seconde}s"
        elif minute > 0:
            string = f"{minute}min {seconde}s"
        else:
            string = f"{seconde}s"

        return string
