import typing as t


class ResultsDict(t.TypedDict):
    train_loss: list[float]
    val_loss: list[float]
