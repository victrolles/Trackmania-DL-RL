import time

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
