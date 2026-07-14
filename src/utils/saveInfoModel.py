import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


from src.utils.logger import get_logger

logger = get_logger("save model info")



def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.info('Model info saved to %s', file_path)
    except Exception as e:
        logger.info('Error occurred while saving the model info: %s', e)
        raise