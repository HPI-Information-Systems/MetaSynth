import pandas as pd

class MLEEvaluator:
    def __init__(self, model, metric, scaler = None):
        self.model = model
        self.metric=metric
        self.scaler = scaler

    def evaluate_default(self, gt, synthetic):
        _, gt_test = gt
        
        if self.scaler is not None:
            gt_test = pd.DataFrame(self.scaler.transform(gt_test), columns=gt_test.columns)
            synthetic = pd.DataFrame(self.scaler.transform(synthetic), columns=synthetic.columns)

        self.model.fit(synthetic[[col for col in synthetic.columns if col != "target"]].to_numpy(), synthetic["target"].astype(int).to_numpy())

        pred_y = self.model.predict(gt_test[[col for col in synthetic.columns if col != "target"]].to_numpy())

        try:
            return self.metric(gt_test["target"].to_numpy(), pred_y, average='macro')
        except:
            return self.metric(gt_test["target"].to_numpy(), pred_y)
