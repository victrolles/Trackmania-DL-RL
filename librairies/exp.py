from librairies.data_classes import RadarState

class Exp:

    def __init__(self, state: RadarState, action: int, reward: float, done: bool, next_state: RadarState):
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done
        self.next_state = next_state
    
    def is_none(self):
        if self.state is None and self.action is None:
            return True
        else:
            return False
        
    def set_none(self):
        self.state = None
        self.action = None
        self.reward = None
        self.done = None
        self.next_state = None