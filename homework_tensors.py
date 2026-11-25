import torch

#1.1

a = torch.rand(3,4)

b = torch.zeros(2,3,4)

c = torch.ones(5,5)

e = torch.arange(16).reshape(4,4)

#1.2

A = torch.arange(12).reshape(3,4)

B = torch.arange(12).reshape(4,3)

print(A.T.shape)
print((A @ B).shape)
print((A * B.T).shape)
print(A.sum())

#1.3

def extract_slices(x: torch.Tensor) -> dict:

    # Проверка входных данных
    if not isinstance(x, torch.Tensor):
        raise TypeError("Ожидался torch.Tensor")
    if x.shape != (5, 5, 5):
        raise ValueError(f"Ожидалась форма (5, 5, 5), получена {tuple(x.shape)}")

    # Трактовка: оси = (лист, строка, столбец).
    # "Первая строка" и "последний столбец" берём на первом листе.
    first_row = x[0, 0]              # лист 0, строка 0 -> форма (5,)
    last_column = x[0, :, -1]        # лист 0, все строки, столбец -1 -> форма (5,)

    # Центр оси длиной 5 это позиция 2. Ровно по центру 2 элемента не уместить,
    # поэтому берём позиции 1..2 (1:3) в центральном листе 2.
    center_2x2 = x[2, 1:3, 1:3]      # форма (2, 2)

    # Чётные индексы 0, 2, 4 по каждой оси -> по 3 элемента на ось.
    even_indices = x[::2, ::2, ::2]  # форма (3, 3, 3)

    return {
        "first_row": first_row,
        "last_column": last_column,
        "center_2x2": center_2x2,
        "even_indices": even_indices,
    }


if __name__ == "__main__":
    # Значения 0..124: элемент X[i, j, k] = 25*i + 5*j + k
    X = torch.arange(125).reshape(5, 5, 5)
    results = extract_slices(X)

    for name, value in results.items():
        print(f"{name}: shape={tuple(value.shape)}")
        print(value, "\n")

#1.4

T = torch.arange(24)

T_2x12 = T.reshape(2,12)

T_3x8 = T.reshape(3,8)

T_4x6 = T.reshape(4,6)

T_2x3x4 = T.reshape(2,3,4)

