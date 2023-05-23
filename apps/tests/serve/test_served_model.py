from ...nn.serve import ServedModel
import torch


class MockModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(MockModel, self).__init__()
        self.fc1 = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.fc1(x)


def test_one_cpu():
    served = ServedModel(MockModel, cpu=1, enable_printing=True)

    result = served.run('__call__', [(torch.randn(10, 10),)], [{}])[0]
    assert result.device == torch.device('cpu')
    assert result.shape == (10, 10)

    result = served.run('forward', [(torch.randn(10, 10),)], [{}])[0]
    assert result.shape == (10, 10)

    result = served('forward', torch.randn(10, 10))
    assert result.shape == (10, 10) 


def test_one_gpu():
    served = ServedModel(MockModel, gpu=1, enable_printing=True)


    result = served.run('__call__', [(torch.randn(10, 10).cuda(),)], [{}])[0]
    assert result.shape == (10, 10)

    result = served.run('forward', [(torch.randn(10, 10).cuda(),)], [{}])[0]
    assert result.shape == (10, 10)


def test_multi_cpu():
    served = ServedModel(MockModel, cpu=2, enable_printing=True)

    result = served.run('__call__', [(torch.randn(10, 10),), (torch.randn(10, 10),), (torch.randn(10, 10),)], [{}, {}, {}])
    assert len(result) == 3
    assert all([x.shape == (10, 10) for x in result])

    result = served.run('forward', [(torch.randn(10, 10),), (torch.randn(10, 10),), (torch.randn(10, 10),)], [{}, {}, {}])
    assert len(result) == 3
    assert all([x.shape == (10, 10) for x in result])



if __name__ == "__main__":
    test_one_cpu()