from ChickenDiseaseClassification.config.configuration import ConfigurationManager
from ChickenDiseaseClassification.components.model_evaluation import Evaluation
from ChickenDiseaseClassification import logger


STAGE_NAME = "Model Evaluation"


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        val_config = config.get_evaluation_config()
        evaluation = Evaluation(val_config)
        evaluation.evaluation()
        evaluation.save_score()
