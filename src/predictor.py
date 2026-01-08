class Predictor:

    def __init__(self, pipeline, threshold=0.5):
        self.pipeline = pipeline
        self.threshold = threshold

    def predict(self, X):
        proba = self.pipeline.predict_proba(X)[:, 1]
        return (proba > self.threshold).astype(int)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)[:, 1]
