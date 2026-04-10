"""Prediction interface for speech emotion recognition."""

import os
import numpy as np
from tensorflow.keras.models import load_model

from src.config import CANONICAL_EMOTIONS, FeatureConfig, MODEL_PATH
from src.feature_extraction import extract_features


class EmotionPredictor:
    """Class for making emotion predictions on audio files."""

    def __init__(self, model_path: str = None):
        """
        Initialize predictor with trained model.

        Args:
            model_path: Path to saved model (uses best model if None)
        """
        if model_path is None:
            model_path = self._find_best_model()

        print(f"Loading model from: {model_path}")
        self.model = load_model(model_path)
        self.emotion_names = list(CANONICAL_EMOTIONS)
        self.feature_config = self._infer_feature_config()

    def _infer_feature_config(self) -> FeatureConfig:
        """
        Infer feature configuration from model input dimensions.

        Legacy MFCC models use 40 feature bins. Robust models use a larger
        stacked feature bundle.
        """
        feature_bins = int(self.model.input_shape[-1])
        if feature_bins == 40:
            return FeatureConfig(
                include_delta=False,
                include_delta2=False,
                include_logmel=False,
                include_zcr=False,
                normalize_per_sample=False,
            )
        return FeatureConfig(
            include_mfcc=True,
            include_delta=True,
            include_delta2=True,
            include_logmel=True,
            include_zcr=True,
            normalize_per_sample=True,
        )

    def _find_best_model(self) -> str:
        """Find the best saved model."""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")

        # Look for best model first
        model_files = [f for f in os.listdir(MODEL_PATH) if f.endswith('_best.keras')]
        if not model_files:
            model_files = [f for f in os.listdir(MODEL_PATH) if f.endswith('.keras')]

        if not model_files:
            raise FileNotFoundError(f"No model found in {MODEL_PATH}")

        return os.path.join(MODEL_PATH, sorted(model_files)[-1])

    def predict(self, audio_path: str) -> dict:
        """
        Predict emotion from audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with prediction results
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Extract features
        features = extract_features(audio_path, feature_config=self.feature_config)

        # Add batch dimension
        features = np.expand_dims(features, axis=0)

        # Get prediction
        probabilities = self.model.predict(features, verbose=0)[0]
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]

        return {
            'emotion': self.emotion_names[predicted_class],
            'confidence': float(confidence),
            'probabilities': {
                name: float(prob)
                for name, prob in zip(self.emotion_names, probabilities)
            }
        }

    def predict_batch(self, audio_paths: list) -> list:
        """
        Predict emotions for multiple audio files.

        Args:
            audio_paths: List of paths to audio files

        Returns:
            List of prediction dictionaries
        """
        results = []
        for path in audio_paths:
            try:
                result = self.predict(path)
                result['file'] = path
                results.append(result)
            except Exception as e:
                results.append({
                    'file': path,
                    'error': str(e)
                })
        return results


def predict_emotion(audio_path: str, model_path: str = None) -> dict:
    """
    Convenience function for single prediction.

    Args:
        audio_path: Path to audio file
        model_path: Path to saved model (optional)

    Returns:
        Prediction result dictionary
    """
    predictor = EmotionPredictor(model_path)
    return predictor.predict(audio_path)


def format_prediction(result: dict) -> str:
    """
    Format prediction result as human-readable string.

    Args:
        result: Prediction result dictionary

    Returns:
        Formatted string
    """
    if 'error' in result:
        return f"Error: {result['error']}"

    output = f"Predicted: {result['emotion']} (confidence: {result['confidence']*100:.1f}%)\n"
    output += "\nAll probabilities:\n"

    # Sort by probability
    sorted_probs = sorted(
        result['probabilities'].items(),
        key=lambda x: x[1],
        reverse=True
    )

    for emotion, prob in sorted_probs:
        bar_len = int(prob * 30)
        bar = '█' * bar_len + '░' * (30 - bar_len)
        output += f"  {emotion:10s} {bar} {prob*100:5.1f}%\n"

    return output


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.predict <audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]
    result = predict_emotion(audio_path)
    print(format_prediction(result))
