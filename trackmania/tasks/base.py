class BaseTask:
    def __init__(self, config=None):
        self.config = config

    def run(self):
        print("BaseTask running with config:", self.config)