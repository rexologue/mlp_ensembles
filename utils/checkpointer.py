import os
from utils.common_functions import read_file, write_file

class Checkpointer:
    def __init__(self, path_to_directory: str, keep_last_n: int = 5):
        self.path = path_to_directory
        os.makedirs(self.path, exist_ok=True)
        
        self.keep_last_n = keep_last_n
        
        
    def save(self, 
             model_state: dict, 
             epoch: int,
             best_metric: float, 
             best: bool = False):
        
        # Call cleaner to remove old checkpoints
        self.cleanup_checkpoints()
        
        filename = f"checkpoint_{epoch}.pickle" if not best else "best_checkpoint.pickle"
        
        dict_to_save = {
            'model_state': model_state,
            'best_metric': best_metric,
            'epoch': epoch
        }
        
        write_file(dict_to_save, os.path.join(self.path, filename))


    def load(self, model, filepath: str):
        checkpoint = read_file(os.path.join(self.path, filepath))
        model.load_params(checkpoint['model_state'])

        return model, checkpoint['epoch'], checkpoint['best_metric']
        
        
    def load_best_model(self, model):
        checkpoint = read_file(os.path.join(self.path, 'best_checkpoint.pickle'))
        model.load_params(checkpoint['model_state'])

        return model

    
    def cleanup_checkpoints(self):
        # Get list of all common checkpoints (without best_checkpoint.pickle)
        checkpoints = os.listdir(self.path)
        checkpoints = [os.path.join(self.path, x) for x in checkpoints if x != 'best_checkpoint.pickle']
        
        # Sort them in ascending order
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[-1].split('.')[0]))
        
        # If amount of checkpoints are larger than keep_last_n - remove old ones
        if len(checkpoints) > self.keep_last_n:
            to_delete = checkpoints[:len(checkpoints) - self.keep_last_n]
            
            for ckpt in to_delete:
                os.remove(ckpt)