import os

def checkpoint_path_corrector(checkpoint_path: str, sub_folder: str = None) -> str:
    """
    Corrects and validates model checkpoint paths similar to HuggingFace structure.
    
    Args:
        checkpoint_path (str): Path to checkpoint file or directory
        sub_folder (str, optional): Sub-directory name if checkpoint is in nested folder
    
    Returns:
        str: Path to the checkpoint file
        
    Raises:
        ValueError: If path doesn't exist or multiple/no checkpoint files found
    """
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path {checkpoint_path} does not exist")
    
    # If it's already a file, return it directly
    if os.path.isfile(checkpoint_path):
        return checkpoint_path
    
    # Handle directory case
    if os.path.isdir(checkpoint_path):
        # Add sub_folder to path if provided
        if sub_folder is not None:
            checkpoint_path = os.path.join(checkpoint_path, sub_folder)
            if not os.path.exists(checkpoint_path):
                raise ValueError(f"Sub-folder path {checkpoint_path} does not exist")
        
        # Find checkpoint files
        files = [f for f in os.listdir(checkpoint_path) 
                if f.endswith((".pth", ".ckpt", ".safetensor", ".safetensors"))]
        
        if not files:
            raise ValueError(f"No checkpoint files found in {checkpoint_path}")
        
        if len(files) > 1:
            raise ValueError(f"Multiple checkpoint files found in {checkpoint_path}: {files}. Please specify the exact file path.")
        
        return os.path.join(checkpoint_path, files[0])
    
    # This line shouldn't be reached due to earlier checks
    raise ValueError(f"Invalid path type for {checkpoint_path}")



def _auto_indexed_filename(filename, dirname):
    suffix = filename.split(".")[-1]

    files = [f for f in os.listdir(dirname) if f.endswith(f".{suffix}")]

    max_index = 0
    if filename in files:
        for f in files:
            f = f.split(".")[0]
            index = int(f.split("_")[-1])
            if index > max_index:
                max_index = index
        
        filename = filename.split(".")[0] + f"_{max_index+1}.{suffix}"
    
    return filename



def create_destination_path(checkpoint_path:str, sub_folder:str = None) ->str:
    try:
        
        ## if given path is a file, return it directly   
        if os.path.isfile(checkpoint_path):
            dirname = os.path.dirname(checkpoint_path)
            ## if dir is not present, create it        
            if not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)

            filename = os.path.basename(checkpoint_path)

            filename = _auto_indexed_filename(filename, dirname)

            checkpoint_path = os.path.join(dirname, filename)
            return checkpoint_path

        else:
            ## if given path is a directory, check if sub_folder is present
            if sub_folder is not None:
                split = os.path.split(checkpoint_path)
            
                if split[-1] != sub_folder:
                    checkpoint_path = os.path.join(checkpoint_path, sub_folder)

            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path, exist_ok=True)


            filename = "model.ckpt"

            filename = _auto_indexed_filename(filename, checkpoint_path)

            checkpoint_path = os.path.join(checkpoint_path, filename)

            return checkpoint_path
        
    except Exception as e:
        print(e)
        return None
