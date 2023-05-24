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
            *args,
            **kwargs
        )

    def run(self, data_source: str, *args, **kwargs):
        return  self._Model.predict(data_source, *args, **kwargs)
    
