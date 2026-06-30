import os, sys
import shutil
import subprocess
import zipfile
import yaml
from objectDetection.utils.main_utils import read_yaml_file
from objectDetection.logger import logging
from objectDetection.exception import SignException
from objectDetection.entity.config_entity import ModelTrainerConfig
from objectDetection.entity.artifacts_entity import ModelTrainerArtifact


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.model_trainer_config = model_trainer_config


    
    def initiate_model_trainer(self,) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            logging.info("Unzipping data")
            with zipfile.ZipFile("object_detection_dataset.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
            os.remove("object_detection_dataset.zip")
           
            with open("data.yaml", 'r') as stream:
                num_classes = str(yaml.safe_load(stream)['nc'])

            model_config_file_name = self.model_trainer_config.weight_name.split(".")[0]
            print(model_config_file_name)

            config = read_yaml_file(f"yolov5/models/{model_config_file_name}.yaml")

            config['nc'] = int(num_classes)


            with open(f'yolov5/models/custom_{model_config_file_name}.yaml', 'w') as f:
                yaml.dump(config, f)

            # Run training using subprocess instead of os.system (cross-platform)
            result = subprocess.run(
                [
                    sys.executable, "train.py",
                    "--img", "416",
                    "--batch", str(self.model_trainer_config.batch_size),
                    "--epochs", str(self.model_trainer_config.no_epochs),
                    "--data", "../object_detection_dataset/data.yaml",
                    "--cfg", f"./models/custom_yolov5s.yaml",
                    "--weights", self.model_trainer_config.weight_name,
                    "--name", "yolov5s_results",
                    "--cache",
                ],
                cwd="yolov5",
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logging.error(f"Training failed: {result.stderr}")
                raise Exception(f"Training process failed with return code {result.returncode}")

            # Copy best weights using shutil (cross-platform)
            best_weights_src = os.path.join("yolov5", "runs", "train", "yolov5s_results", "weights", "best.pt")
            shutil.copy2(best_weights_src, "yolov5/")
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            shutil.copy2(best_weights_src, self.model_trainer_config.model_trainer_dir)
           
            # Cleanup using shutil (cross-platform)
            shutil.rmtree("yolov5/runs", ignore_errors=True)
            shutil.rmtree("train", ignore_errors=True)
            shutil.rmtree("test", ignore_errors=True)
            if os.path.exists("data.yaml"):
                os.remove("data.yaml")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path="yolov5/best.pt",
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact


        except Exception as e:
            raise SignException(e, sys)