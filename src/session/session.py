class Session:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.selected_features = None
        self.models = {} 
        self.optimised_models = {} 
        self.model = None