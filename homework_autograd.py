import torch

#2.1

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = torch.tensor(4.0, requires_grad=True)

f = x**2 + y**2 + z**2 + 2*x*y*z

f.backward()
print(x.grad)
print(y.grad)
print(z.grad)

#2.2
import torch.nn.functional as F

def mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    #Среднеквадратичная ошибка: (1/n) * sum((y_pred - y_true)^2).

    if not isinstance(y_pred, torch.Tensor) or not isinstance(y_true, torch.Tensor):
        raise TypeError("Оба аргумента должны быть torch.Tensor")
    if not (y_pred.is_floating_point() and y_true.is_floating_point()):
        raise TypeError("Тензоры должны быть дробного типа (float)")
    # Без этой проверки broadcasting молча превратил бы формы (4,) и (4, 1)
    # в матрицу (4, 4), и MSE получился бы неверным
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"Формы не совпадают: {tuple(y_pred.shape)} и {tuple(y_true.shape)}"
        )
    n = y_pred.numel()
    if n == 0:
        raise ValueError("Пустые тензоры")

    errors = y_pred - y_true
    return (errors ** 2).sum() / n


def task_2_2() -> None:
    """Задание 2.2: градиенты MSE по параметрам w и b линейной модели."""
    # Данные: фиксированные, градиенты по ним не нужны
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_true = torch.tensor([3.0, 5.0, 7.0, 9.0])

    # Параметры модели: по ним нужны градиенты
    w = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    # Прямой проход: строится граф вычислений
    y_pred = w * x + b
    loss = mse(y_pred, y_true)

    # Обратный проход: заполняются w.grad и b.grad
    loss.backward()
    assert w.grad is not None and b.grad is not None, \
        "backward() не заполнил градиенты"

    print(f"MSE:      {loss.item()}")
    print(f"dMSE/dw:  {w.grad.item()}")
    print(f"dMSE/db:  {b.grad.item()}")

    # Проверка 1: ручные формулы
    #   dMSE/dw = (2/n) * sum(u * x),  dMSE/db = (2/n) * sum(u),
    #   где u = y_pred - y_true
    n = x.numel()
    u = y_pred.detach() - y_true            # detach: расчёт вне графа
    expected_w = (2 / n) * (u * x).sum()
    expected_b = (2 / n) * u.sum()
    assert torch.allclose(w.grad, expected_w), "grad w не совпал с формулой"
    assert torch.allclose(b.grad, expected_b), "grad b не совпал с формулой"

    # Проверка 2: эталонная реализация из библиотеки
    assert torch.allclose(loss.detach(), F.mse_loss(y_pred.detach(), y_true))

    # Проверка 3: числа, посчитанные на бумаге
    assert torch.allclose(loss.detach(), torch.tensor(13.5))
    assert torch.allclose(w.grad, torch.tensor(-20.0))
    assert torch.allclose(b.grad, torch.tensor(-7.0))

    # Проверка 4: обработка ошибок
    for bad_args in [
        (torch.ones(4), torch.ones(3)),       # разные формы
        (torch.ones(4), torch.ones(4, 1)),    # ловушка broadcasting
    ]:
        try:
            mse(*bad_args)
        except ValueError as e:
            print("Ожидаемая ошибка:", e)
        else:
            raise AssertionError("ValueError не был вызван")

    try:
        mse(torch.tensor([1, 2]), torch.tensor([1, 2]))   # целые числа
    except TypeError as e:
        print("Ожидаемая ошибка:", e)
    else:
        raise AssertionError("TypeError не был вызван")

    print("Все проверки пройдены")


if __name__ == "__main__":
    task_2_2()

#2.3

x = torch.tensor(1.0, requires_grad=True)

f = torch.sin(x**2 + 1)

(df_dx,) = torch.autograd.grad(f, x)

print(df_dx)