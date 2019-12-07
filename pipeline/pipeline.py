

class Pipeline:
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.model = None

    def load_data(self):
        self.train_data = None
        self.test_data = None

    def build_model(self):
        self.model = Model()
        self.model.build()

    def train_model(self):
        self.model.train(self.train_data)

    def predict(self):
        self.model.predict(self.test_data)

    def publish_outputs(self):
        self.model.publish_outputs()

