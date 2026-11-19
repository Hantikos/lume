# Руководство по Lume за 20 минут

Полный туториал: от первой строки до собранного приложения.

---

## Шаг 0. Установка

```bash
git clone https://github.com/Hantikos/lume.git
cd lume
python -m lume examples
```

Нужен Python 3.9 или новее. Больше ничего устанавливать не нужно.

---

## Шаг 1. Первая программа

Создайте `hello.lume`:

```lume
print "Привет, Lume!"
```

Запуск:

```bash
python -m lume run hello.lume
```

Вывод:

```
Привет, Lume!
```

---

## Шаг 2. Переменные и строки

```lume
let name = "Аня"
let age = 21
let happy = true

print f"{name}, тебе {age} лет"
print "счастлива? " + happy
```

`f"..."` — интерполяция: внутрь `{}` можно класть любое выражение.

---

## Шаг 3. Функции

```lume
fn area(width, height) {
    return width * height
}

print area(3, 4)          # 12
```

Значения по умолчанию:

```lume
fn greet(name, greeting = "Здравствуйте") {
    print f"{greeting}, {name}!"
}
greet("Мир")
greet("Мир", "Привет")
```

---

## Шаг 4. Списки, словари, циклы

```lume
let fruits = ["яблоко", "банан", "вишня"]
for fruit in fruits {
    print fruit
}

let prices = {"яблоко": 120, "банан": 90}
print prices["яблоко"]

let total = 0
for price in values(prices) {
    total = total + price
}
print total
```

---

## Шаг 5. Условия

```lume
let score = 87

if score >= 90 {
    print "отлично"
} else if score >= 70 {
    print "хорошо"
} else {
    print "нужно повторить"
}
```

---

## Шаг 6. Данные и первая модель

```lume
data points = dataset.xor()

model XorNet {
    layer Dense(16, "relu")
    layer Dense(1, "sigmoid")

    train(
        data = points,
        epochs = 150,
        lr = 0.2,
        optimizer = "adam",
        loss = "binary_crossentropy"
    )

    eval(data = points)
}

print predict(XorNet, [[1, 1]])   # 0 — XOR
```

Что происходит внутри:

1. `dataset.xor()` возвращает 4 точки с метками.
2. `model` создаёт сеть: 16 нейронов `relu`, затем 1 нейрон `sigmoid`.
3. `train` подбирает веса (обратное распространение ошибки вручную, без numpy).
4. `eval` печатает метрики и матрицу ошибок.
5. `predict` возвращает класс.

---

## Шаг 7. Свой датасет

```lume
# список из 4 точек: x и метка
data my = dataset([[0.2, 0.1], [0.9, 0.8], [0.4, 0.2], [1.0, 0.9]], [0, 1, 0, 1])

model MyNet {
    layer Dense(8, "relu")
    layer Dense(2, "softmax")

    train(data = my, epochs = 100, lr = 0.1, optimizer = "adam", loss = "crossentropy")
    eval(data = my)
}
```

CSV-файл:

```lume
data from_csv = dataset.csv("data.csv")
```

Формат файла: последняя колонка — метка, остальные — признаки.

---

## Шаг 8. Регрессия

Для предсказания числа нужен «сырой» вывод сети — `forward`:

```lume
data points = dataset.linear(count = 60, noise = 0.2)

model Line {
    layer Dense(16, "relu")
    layer Dense(1, "none")

    train(data = points, epochs = 200, lr = 0.02, optimizer = "adam", loss = "mse")
}

print forward(Line, [[1.0]])   # близко к 5 (прямая y = 3x + 2)
```

---

## Шаг 9. Работа с Lume Studio

```bash
python -m lumeide
```

* Слева — файлы проекта и готовые примеры.
* В центре — редактор с подсветкой; `Ctrl+Enter` запускает код, результат в консоли.
* Справа — ИИ-ассистент. Спросите «объясни этот код» или «напиши модель для
  классификации» — он отвечает прямо в редакторе, зная ваш текущий файл.
* «Собрать ▾» → **Модель** / **Приложение** / **EXE**.

---

## Шаг 10. Сборка приложения

```bash
python -m lume build app examples/xor.lume -o build
cd build/XorNetApp
python app.py
```

Откройте <http://127.0.0.1:8000> — там форма для предсказаний.
Из консоли: `python app.py --predict "1,0"`.

---

## Шаг 11. Сборка EXE

```bash
pip install pyinstaller
python tools/build_exe.py --all
```

На Windows получится `dist/LumeStudio.exe` и `dist/lume.exe` — одиночные файлы,
которые можно просто скопировать на другой компьютер.

---

## Шаг 12. Картинки и свёрточная сеть

```lume
data images = dataset.bars(count = 60, size = 8)

model LineNet {
    layer Conv2d(4, 3, 8, 8, 1, "relu")   # 4 фильтра 3x3
    layer MaxPool(2)                      # 8x8 -> 3x3
    layer Flatten()
    layer Dense(16, "relu")
    layer Dense(2, "softmax")

    train(data = images, epochs = 40, lr = 0.02, optimizer = "adam", loss = "crossentropy")
    eval(data = images)
}
```

Размер картинки задаётся в `Conv2d` (`height`, `width`, `channels`),
`MaxPool` наследует размеры сам. Обучение такой сети на чистом Python
занимает пару секунд.

---

## Шаг 13. Куда двигаться дальше

* `examples/tour.lume` — полный обзор возможностей языка.
* `examples/text_sentiment.lume` — классификатор текста на bag-of-words.
* `docs/LANGUAGE.md` — спецификация.
* `lume repl` — эксперименты прямо в терминале.
