import torch
import ultralytics

class YOLOv8Wrapper:
    def __init__(
            self,
            pt_path: str,
            *args,
            **kwargs
            
        ) -> None:
    
        self._Model = ultralytics.YOLO(
            pt_path,
            task='detect',
            *args,
            **kwargs
        )

    def run(self, data_source: str, *args, **kwargs):
        device = 'cpu'
        if torch.cuda.is_available() and hasattr(self, '_ShouldUseGPU'):
            device = 0

        return  self._Model.predict(data_source, *args, **kwargs, device=device)
    
