import numpy as np

class Ensemble():
    def __init__(self, models: list, model_name_list:list):
        self.models = models
        self.model_name_lst = model_name_list

    def predict(self, X, weights, threshold):
        # Get predictions -- could be parallelised
        outputs, output_dict = self.get_model_preds(X)

        # Reformat predictions to put all model's predictions for one row together
        prediction_compiled_lst = self.get_prediction_compiled_list(X, outputs)
        # Apply weights to predictions to get weighted average
        yhats = np.array(prediction_compiled_lst)
        avg = np.average(yhats, axis=1, weights=weights)

        # Convert probabilities to 0 or 1 classification outcome
        discrete = (avg > threshold).astype("int32")

        pred = "AI" if discrete[0] == 1 else "Human"
        return pred, output_dict

    # Reformat predictions such that each list contains predictions from all models
    def get_prediction_compiled_list(self, X, outputs):
        prediction_compiled_lst = []
        for i in range(len(X)):
            prediction_lst = []
            for output in outputs:
                prediction_lst.append(output[i])
            prediction_compiled_lst.append(prediction_lst)
        return prediction_compiled_lst

    def get_model_preds(self, X):
        # Get predictions -- could be parallelised
        outputs = []
        output_dict = {}
        for i in range(len(self.models)):
            model = self.models[i]
            output = model.predict(X)
            outputs.append(output)
            output_dict[self.model_name_lst[i]] = output

        return outputs, output_dict
