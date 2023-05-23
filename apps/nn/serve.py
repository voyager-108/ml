import logging
from subprocess import PIPE, Popen
from typing import Any, Callable
from xml.etree.ElementTree import fromstring

from rich.console import Console  
from rich.table import Table
from concurrent.futures import Future, ProcessPoolExecutor

def init_model_worker(constructor, args, kwargs, _UseGPU: bool = False):
    global _Model
    _Model = constructor(*args, **kwargs)

    if hasattr(_Model, 'to') and _UseGPU:
        try:  _Model.to('cuda')
        except: ...


def run_model_worker(attr: str, *args, **kwargs):
    _X = getattr(_Model, attr)(*args, **kwargs)

    if hasattr(_X, 'detach'):
        _X = getattr(_X, 'detach')()
    
    if hasattr(_X, 'cpu'):
        _X = getattr(_X, 'cpu')()
    
    return _X

def get_gpu_info(*args, **kwargs):
    p = Popen(["nvidia-smi", "-q", "-x"], stdout=PIPE)
    outs, errors = p.communicate()
    xml = fromstring(outs)
    datas = []
    driver_version = xml.findall("driver_version")[0].text
    cuda_version = xml.findall("cuda_version")[0].text

    for gpu_id, gpu in enumerate(xml.iterfind("gpu")):
        gpu_data = {}
        name = [x for x in gpu.iterfind("product_name")][0].text
        memory_usage = gpu.findall("fb_memory_usage")[0]
        total_memory = memory_usage.findall("total")[0].text

        gpu_data["name"] = name
        gpu_data["total_memory"] = total_memory
        gpu_data["driver_version"] = driver_version
        gpu_data["cuda_version"] = cuda_version
        datas.append(gpu_data)
    return datas 





class ServedModel:
    def __init__(
            self, 
            model: Callable,
            *model_args,
            cpu: int = None,
            gpu: int = None,
            enable_logging=False,
            enable_printing=False,
            logging_level=logging.INFO,
            **model_kwargs,
        ) -> None:
        
        self._ModelName = model.__name__

        self._EnableLogging = enable_logging
        self._EnablePrinting = enable_printing

        self._Logger = logging.getLogger(
            f"served::{self._ModelName}"
        )

        if self._EnableLogging:
            self._Logger.setLevel(logging_level)
            self._Logger.handlers.clear()
            self._Logger.addHandler(logging.StreamHandler())

        
        if self._EnablePrinting:
            self._RichConsole = Console()


        if cpu is None and gpu is None:
            raise ValueError("Either cpu or gpu must be specified")
        
        if self._EnablePrinting:
            self._RichConsole.print(f"[bold]Serving [green]{self._ModelName}[/green][/bold]")
    


        _StatTable = Table(show_header=False)
        _StatTable.add_row("CPU", str(cpu))
        _GPUData = get_gpu_info()
        for i, _GPU in enumerate(_GPUData):
            _StatTable.add_row(f"GPU {i}", f"{_GPU['name']} ({_GPU['total_memory']})")

        if self._EnablePrinting:
            self._RichConsole.print(_StatTable)

        self._Worker = ProcessPoolExecutor(
            max_workers=cpu or 1,
            initializer=init_model_worker,
            initargs=(model, model_args, model_kwargs, gpu is not None),
        )

        if self._EnablePrinting:
            self._RichConsole.print(f"[bold]Started serving [green]{self._ModelName}[/green][/bold]")

    def __Call_worker(self, fn: str=None, *args, **kwargs) -> Any:
        fn = fn or '__call__'

        _ArgsString = str(args)
        if len(_ArgsString) > 20:
            _ArgsString = _ArgsString[:20] + '...'
        
        _KwargsString = str(kwargs)
        if len(_KwargsString) > 20:
            _KwargsString = _KwargsString[:20] + '...'
        
        if self._EnablePrinting:
            self._RichConsole.print(f"- Calling [bold magenta]{fn}[/bold magenta] with args: \"{_ArgsString}\" and kwargs: {_KwargsString}")
        return self._Worker.submit(run_model_worker, fn, *args, **kwargs)
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if self._EnablePrinting:
            self._RichConsole.print(f"Single call:")
        return self.__Call_worker(*args, **kwds).result()
    

    def run(self, fn: str, args: list[Any] = [], kwargs: list[dict[Any, Any]] = []) -> Any:
        fn = fn or '__call__'
        if self._EnablePrinting:
            self._RichConsole.print(f"Run [bold dark_orange]{fn}[/bold dark_orange]:")
        results: list[Future] = []
        for arg, kwarg in zip(args, kwargs):
            results.append(
                self.__Call_worker(fn, *arg, **dict(kwarg))
            )
        return [r.result() for r in results]
    

    def __del__(self) -> None:
        if hasattr(self, '_Worker'):
            self._Worker.shutdown()
        if self._EnablePrinting:
            self._RichConsole.print(f"[bold]Stopped serving [green]{self._ModelName}[/green][/bold]")

    