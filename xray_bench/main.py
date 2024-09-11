import time
from typing import List, Dict
from datasets import load_dataset
from pydantic import BaseModel
from sklearn.metrics import accuracy_score, precision_score, recall_score
from loguru import logger
import cv2
import random


# Pydantic schema for evaluation metrics
class EvaluationMetrics(BaseModel):
    total_images: int = 0
    correct_predictions: int = 0
    total_time_taken: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0


# Log configuration
logger.add("evaluation_log.log", rotation="500 MB")


# Dummy agent class for anomaly detection
class AnomalyDetectionAgent:
    def __init__(self) -> None:
        self.name = "AnomalyDetectionAgent"

    def run(self, xray_image: cv2.Mat) -> str:
        """
        Processes the X-ray image and returns a predicted label.

        Args:
            xray_image (cv2.Mat): Preprocessed X-ray image.

        Returns:
            str: Predicted label (Emphysema, Pneumothorax).
        """
        predicted_labels = ["Emphysema", "Pneumothorax"]
        return random.choice(predicted_labels)


def preprocess_image(image_path: str) -> cv2.Mat:
    """
    Loads and preprocesses an X-ray image by resizing and normalizing it.

    Args:
        image_path (str): Path to the X-ray image.

    Returns:
        cv2.Mat: Preprocessed image.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224))  # Resize to 224x224
    image = image / 255.0  # Normalize
    return image


class EvaluationSuite:
    def __init__(self) -> None:
        self.metrics = EvaluationMetrics()
        self.ground_truth_labels: List[str] = []
        self.predicted_labels: List[str] = []

    def run_agent_on_dataset(self, agent: callable, dataset: List[Dict]) -> None:
        """
        Runs the agent on every row in the dataset and evaluates the predictions.

        Args:
            agent (AnomalyDetectionAgent): The agent responsible for processing the X-ray images.
            dataset (List[Dict]): The dataset containing X-ray images and labels.
        """
        logger.info(f"Starting evaluation with {agent.name}")

        start_time = time.time()

        for row in dataset:
            image_path = row['image']
            label = row['label']  # Ground truth label

            # Preprocess image
            preprocessed_image = preprocess_image(image_path)

            # Run agent
            prediction = agent.run(preprocessed_image)

            # Log the process
            logger.debug(f"Processed image {image_path} -> Prediction: {prediction}, Label: {label}")

            # Store results for evaluation
            self.ground_truth_labels.append(label)
            self.predicted_labels.append(prediction)

            # Increment correct predictions if label matches
            if prediction == label:
                self.metrics.correct_predictions += 1

            # Track total images processed
            self.metrics.total_images += 1

        end_time = time.time()

        # Calculate time taken
        self.metrics.total_time_taken = end_time - start_time

    def validate(self) -> None:
        """
        Validates the agent's predictions by calculating accuracy, precision, and recall.
        """
        logger.info("Validating predictions...")

        # Calculate evaluation metrics
        self.metrics.accuracy = accuracy_score(self.ground_truth_labels, self.predicted_labels)
        self.metrics.precision = precision_score(self.ground_truth_labels, self.predicted_labels, average='weighted')
        self.metrics.recall = recall_score(self.ground_truth_labels, self.predicted_labels, average='weighted')

        # Log the results
        logger.success(f"Evaluation Complete! Accuracy: {self.metrics.accuracy:.4f}, "
                       f"Precision: {self.metrics.precision:.4f}, Recall: {self.metrics.recall:.4f}")
        logger.info(f"Total images processed: {self.metrics.total_images}")
        logger.info(f"Correct predictions: {self.metrics.correct_predictions}")
        logger.info(f"Total time taken: {self.metrics.total_time_taken:.2f} seconds")

    def save_metrics(self, filepath: str = "eval_metrics.json") -> None:
        """
        Saves the evaluation metrics to a JSON file.

        Args:
            filepath (str): Path to save the metrics.
        """
        with open(filepath, "w") as file:
            file.write(self.metrics.json(indent=4))
        logger.info(f"Evaluation metrics saved to {filepath}")


def run_evaluation():
    """
    Main function to load the dataset, run the agent on it, and evaluate the predictions.
    """
    logger.info("Loading dataset...")
    
    # Load the NIH Chest X-ray 14 dataset (subset for demo purposes)
    dataset = load_dataset("BahaaEldin0/NIH-Chest-Xray-14", split="train").shuffle(seed=42).select(range(100))  # Taking 100 samples

    # Initialize the agent
    agent = AnomalyDetectionAgent()

    # Initialize evaluation suite
    evaluation_suite = EvaluationSuite()

    # Run the agent on every row of the dataset
    evaluation_suite.run_agent_on_dataset(agent, dataset)

    # Validate the predictions and calculate metrics
    evaluation_suite.validate()

    # Save the metrics to a file
    evaluation_suite.save_metrics()


if __name__ == "__main__":
    run_evaluation()
