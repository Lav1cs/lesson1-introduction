import time
from typing import Callable, Optional

import torch

SHAPES = [(64, 1024, 1024), (128, 512, 512), (256, 256, 256)]

WARMUP = 2            # прогрев: первые запуски медленнее (инициализация, кэши)
REPEATS_CPU = 3       # на CPU тяжёлые операции идут секунды, повторов немного
REPEATS_GPU = 10

# Операции из условия. Транспонирование делаем с .contiguous():
# сам transpose() лишь меняет метаданные и данные не копирует (время ~0),
# а contiguous() заставляет реально переложить данные в памяти.
OPERATIONS: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "Матричное умножение": lambda a, b: torch.matmul(a, b),
    "Сложение": lambda a, b: a + b,
    "Умножение": lambda a, b: a * b,
    "Транспонирование": lambda a, b: a.transpose(1, 2).contiguous(),
    "Сумма элементов": lambda a, b: a.sum(),
}


def get_accelerator() -> Optional[str]:
    """Возвращает 'mps' (GPU Apple Silicon) или None, если ускорителя нет."""
    if torch.backends.mps.is_available():
        return "mps"
    return None


def synchronize(device_type: str) -> None:
    """Ждёт завершения всех поставленных в очередь операций на устройстве.

    GPU работает асинхронно: без ожидания замер покажет лишь время
    постановки задачи в очередь, а не время вычисления.
    """
    if device_type == "mps":
        torch.mps.synchronize()


def free_memory(device_type: Optional[str]) -> None:
    """Освобождает кэш памяти ускорителя между размерами."""
    if device_type == "mps":
        torch.mps.empty_cache()


def create_matrices(shape: tuple) -> tuple[torch.Tensor, torch.Tensor]:
    """Создаёт две матрицы заданной формы со случайными числами из [0, 1).

    Args:
        shape: форма (пакет, строки, столбцы).

    Raises:
        ValueError: если форма не трёхмерная или есть неположительные размеры.
    """
    if len(shape) != 3 or any(d <= 0 for d in shape):
        raise ValueError(f"Нужна форма из трёх положительных чисел, получено {shape}")
    return torch.rand(shape), torch.rand(shape)


def measure_time(
    op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    a: torch.Tensor,
    b: torch.Tensor,
    repeats: int,
    warmup: int = WARMUP,
) -> float:
    """Измеряет среднее время выполнения операции в миллисекундах.

    Устройство определяется по тензору `a`. Используется time.time();
    для ускорителя перед остановкой таймера выполняется синхронизация.

    Args:
        op: функция двух тензоров.
        a, b: входные тензоры (уже лежат на нужном устройстве).
        repeats: число замеряемых запусков.
        warmup: число прогревочных запусков (не учитываются).

    Returns:
        Среднее время одного запуска в мс.
    """
    if a.shape != b.shape:
        raise ValueError(f"Формы не совпадают: {tuple(a.shape)} и {tuple(b.shape)}")
    if repeats < 1:
        raise ValueError("repeats должен быть >= 1")

    device_type = a.device.type

    for _ in range(warmup):                 # прогрев
        op(a, b)
    synchronize(device_type)                # убедились, что прогрев закончен

    start_time = time.time()
    for _ in range(repeats):
        op(a, b)
    synchronize(device_type)                # без этого замер на GPU некорректен
    return (time.time() - start_time) * 1000 / repeats   # секунды -> мс


def self_test(accelerator: Optional[str]) -> None:
    """Быстрая проверка: операции верны, результаты CPU и ускорителя совпадают."""
    a, b = create_matrices((2, 8, 8))
    # Простой пример с известным ответом: единичные матрицы
    eye = torch.eye(4).repeat(2, 1, 1)
    assert torch.allclose(OPERATIONS["Матричное умножение"](eye, eye), eye)
    assert OPERATIONS["Транспонирование"](a, b).shape == (2, 8, 8)
    assert torch.allclose(OPERATIONS["Сумма элементов"](eye, eye), torch.tensor(8.0))

    if accelerator is not None:
        a_d, b_d = a.to(accelerator), b.to(accelerator)
        for name, op in OPERATIONS.items():
            cpu_res = op(a, b)
            dev_res = op(a_d, b_d).cpu()
            assert torch.allclose(cpu_res, dev_res, rtol=1e-3, atol=1e-3), \
                f"Результаты CPU и {accelerator} различаются: {name}"
    print("Самопроверка пройдена\n")


def print_table(rows: list, accelerator: Optional[str]) -> None:
    """Печатает таблицу результатов в формате из условия."""
    gpu_name = f"GPU/{accelerator}" if accelerator else "GPU"
    print(f"{'Операция':<22}| {'CPU (мс)':>10} | {gpu_name + ' (мс)':>16} | {'Ускорение':>10}")
    print("-" * 66)
    for name, cpu_ms, gpu_ms in rows:
        if gpu_ms is None:
            gpu_str, speed_str = "N/A", "N/A"
        else:
            gpu_str, speed_str = f"{gpu_ms:.2f}", f"{cpu_ms / gpu_ms:.1f}x"
        print(f"{name:<22}| {cpu_ms:>10.2f} | {gpu_str:>16} | {speed_str:>10}")


def run_benchmark() -> None:
    """Запускает сравнение CPU и ускорителя для всех размеров и операций."""
    accelerator = get_accelerator()
    print(f"Ускоритель: {accelerator or 'не найден, только CPU'}")
    if accelerator == "mps":
        print("Внимание: MPS это не CUDA, цифры отличаются от NVIDIA GPU\n")

    self_test(accelerator)

    for shape in SHAPES:
        size_mb = shape[0] * shape[1] * shape[2] * 4 / 1024**2
        print(f"=== Форма {shape} ({size_mb:.0f} МБ на тензор, float32) ===")
        a_cpu, b_cpu = create_matrices(shape)

        a_dev = b_dev = None
        if accelerator is not None:
            # Перенос данных ДО замера: здесь меряем вычисления, а не копирование
            a_dev, b_dev = a_cpu.to(accelerator), b_cpu.to(accelerator)
            synchronize(accelerator)

        rows = []
        best = (None, 0.0)
        for name, op in OPERATIONS.items():
            cpu_ms = measure_time(op, a_cpu, b_cpu, REPEATS_CPU, warmup=1)
            gpu_ms = None
            if accelerator is not None:
                try:
                    gpu_ms = measure_time(op, a_dev, b_dev, REPEATS_GPU)
                except RuntimeError as err:          # например, нехватка памяти
                    print(f"  {name}: ошибка на {accelerator}: {err}")
            rows.append((name, cpu_ms, gpu_ms))
            if gpu_ms is not None and cpu_ms / gpu_ms > best[1]:
                best = (name, cpu_ms / gpu_ms)

        print_table(rows, accelerator)
        if best[0] is not None:
            print(f"Наибольшее ускорение: {best[0]} ({best[1]:.1f}x)")
        print()

        # Освобождаем память перед следующим размером
        del a_cpu, b_cpu, a_dev, b_dev
        free_memory(accelerator)


if __name__ == "__main__":
    run_benchmark()