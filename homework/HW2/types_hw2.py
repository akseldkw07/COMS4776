import typing as t


class ResultsDict(t.TypedDict):
    train_loss: list[float]
    val_loss: list[float]


class TrainerHistDict(t.TypedDict):
    train_loss: list[float]
    val_loss: list[float]
    flops: list[int]
    tokens: list[int]
