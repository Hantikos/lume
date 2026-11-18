# Спецификация языка Lume / Lume language specification

Версия: **0.1** · Язык реализован на чистом Python (стандартная библиотека).

---

## 1. Лексика

### 1.1 Комментарии

```
# это комментарий до конца строки
```

### 1.2 Литералы

| Тип | Примеры |
| --- | --- |
| Целое число | `0`, `42`, `1_000` |
| Дробное число | `3.14`, `.5`, `1e-3`, `2.5e10` |
| Строка | `"текст"`, `'текст'` |
| Интерполируемая строка | `f"Привет, {name}!"` |
| Список | `[1, 2, 3]`, `[]` |
| Словарь | `{"a": 1, "b": 2}` |
| Логические | `true`, `false` |
| Пустое значение | `null` |

Управляющие символы в строках: `\n`, `\t`, `\r`, `\\`, `\"`, `\'`, `\0`, `\uXXXX`.

### 1.3 Ключевые слова

```
fn  let  const  if  else  for  in  while  return  break  continue
import  from  as  true  false  null  and  or  not
```

Слова `model`, `layer`, `data` работают как ключевые слова только в своей позиции,
поэтому их можно использовать и как имена переменных.

### 1.4 Операторы

| Оператор | Значение |
| --- | --- |
| `+ - * / % **` | арифметика (`**` — степень) |
| `== != < <= > >=` | сравнения |
| `and or not` | логика |
| `=` | присваивание |
| `@` | умение матриц (тензоров) |
| `+` со строкой | конкатенация: `"x = " + 5` → `"x = 5"` |

---

## 2. Структура программы

Программа — это последовательность утверждений. Блоки задаются фигурными скобками,
`;` в конце утверждения необязателен.

```lume
fn main() {
    print "hi"
}
```

### 2.1 Переменные

```lume
let x = 10          # изменяемая
const limit = 100   # константа
x = x + 1
items[0] = 5        # присваивание по индексу
table["key"] = 7    # присваивание в словарь
```

`let` создаёт имя в текущей области видимости; обычное `=` переприсваивает
существующее имя (ищется вверх по цепочке областей).

### 2.2 Функции

```lume
fn add(a, b) {
    return a + b
}

fn greet(name, greeting = "hi") {   # значение по умолчанию
    print f"{greeting}, {name}!"
}

fn fact(n) {                        # рекурсия разрешена
    if n <= 1 { return 1 }
    return n * fact(n - 1)
}
```

Функции — значения: их можно передавать и возвращать, они замыкают внешнюю область.

### 2.3 Условия и циклы

```lume
if x > 10 {
    print "big"
} else if x > 3 {
    print "medium"
} else {
    print "small"
}

for item in list {
    if item == 0 { continue }
    if item < 0 { break }
    print item
}

while counter < 10 {
    counter = counter + 1
}
```

`for` умеет итерироваться по спискам, строкам, словарям (по ключам), диапазонам
`range(...)` и тензорам.

### 2.4 Оператор `print`

```lume
print "обычный текст"
print "число:", 42
print f"интерполяция: {1 + 2}"
print([1, 2, 3])       # для списков скобки обязательны
```

---

## 3. Модели

Модель — это именованный блок слоёв, который можно обучить прямо в коде.

```lume
model MyNet {
    layer Dense(128, "relu")
    layer Dropout(0.2)
    layer Dense(10, "softmax")

    train(data = dataset, epochs = 20, lr = 0.001, optimizer = "adam", loss = "crossentropy")
    eval(data = dataset)
    save("my_model.json")
}
```

### 3.1 Слои

| Слой | Вид | Параметры |
| --- | --- | --- |
| `Dense` | обучаемый | `Dense(units, activation)` |
| `Activation` | — | `Activation("relu")` |
| `Dropout` | — | `Dropout(rate)` |
| `Flatten` | — | `Flatten()` |
| `Embedding` | обучаемый | `Embedding(vocab_size, dim)` |

Активации: `none`, `linear`, `relu`, `sigmoid`, `tanh`, `gelu`, `softmax`.

### 3.2 Данные

```lume
data points = dataset.xor()                       # встроенные наборы
data raw    = dataset.csv("data.csv")             # CSV-файл
data custom = dataset([[0, 0], [0, 1]], [0, 1])   # вручную: x и метки
data tensor_data = dataset(tensor([[1, 2]]), [1])
```

Готовые генераторы:

| Вызов | Что делает |
| --- | --- |
| `dataset.xor()` | 4 точки XOR, 2 класса |
| `dataset.linear(count, noise, seed)` | случайные точки на прямой `y = 3x + 2` |
| `dataset.blobs(classes, per_class, seed)` | gaussian-кластеры |
| `dataset.spiral(count, noise, seed)` | спираль |

### 3.3 Обучение

```lume
train(
    data = points,
    epochs = 50,
    lr = 0.01,
    optimizer = "adam",
    loss = "binary_crossentropy",
    batch_size = 16,
    verbose = true
)
```

* оптимизаторы: `sgd`, `momentum`, `rmsprop`, `adam`
* лоссы: `mse`, `mae`, `binary_crossentropy`, `crossentropy` (softmax + индексы классов)
* `batch_size` по умолчанию 16

### 3.4 Оценка и предсказание

```lume
eval(data = points)            # печатает accuracy, precision, recall, f1 и матрицу ошибок
predict(MyNet, [[1, 0]])       # классы: [1]
forward(MyNet, [[1, 0]])       # сырой вывод сети: [0.97]
predict_proba(MyNet, [[1, 0]]) # вероятности
```

`train`, `eval`, `predict`, `forward`, `save` и `load` внутри блока `model`
автоматически работают с текущей моделью — передавать её первым аргументом не нужно.

### 3.5 Сохранение

```lume
save("model.json")     # веса + архитектура в JSON
let other = load_model("model.json")
```

Формат файла:

```json
{
  "format": "lume-model/1",
  "name": "MyNet",
  "layers": [
    {"type": "Dense", "units": 16, "activation": "relu", "W": [[...]], "b": [...]}
  ],
  "metrics": {"accuracy": 1.0}
}
```

---

## 4. Тензоры

```lume
let t = tensor([[1, 2], [3, 4]])
print t.shape          # [2, 2]
print t @ tensor([[1], [1]])   # [[3], [7]]
print zeros(3, 2)
print ones(2, 2)
print randn(4, 3, seed = 42)
```

Тензор — это матрица `rows × cols` из чисел с плавающей точкой.

---

## 5. Встроенные функции

**Вывод и ввод:** `print`, `println`, `input`

**Типы:** `type`, `str`, `int`, `float`, `bool`, `list`, `dict`, `len`, `keys`, `values`

**Математика:** `sum`, `min`, `max`, `abs`, `round`, `floor`, `ceil`, `sqrt`, `exp`,
`log`, `log2`, `sin`, `cos`, `tan`, `pow`, `mean`, `median`, `argmax`, `softmax`,
`sigmoid`, `relu`

**Списки:** `range`, `sort`, `shuffle`, `unique`, `count`, `seed`

**Строки:** `upper`, `lower`, `strip`, `split`, `join`, `replace`, `startswith`,
`endswith`, `contains`

**Данные и модели:** `tensor`, `zeros`, `ones`, `randn`, `eye`, `transpose`,
`dataset`, `read_csv`, `write_csv`, `train`, `eval`, `evaluate`, `predict`,
`predict_proba`, `forward`, `save`, `load`, `save_model`, `load_model`,
`confusion_matrix`

**Модули:** `import math`, `import json`, `import lume`

---

## 6. Отступы и комментарии в коде

Отступы в Lume не влияют на выполнение (блоки задаются скобками), но код читается
как обычно — по четыре пробела.

---

## 7. Ошибки и коды возврата

Ошибки времени выполнения содержат номер строки и колонки:

```
Line 12:7  unknown name 'points'
```

`lume check file.lume` только проверяет синтаксис и возвращает код `0` или `1`.

---

## 8. Совместимость

* Язык и студия используют только стандартную библиотеку Python 3.9+.
* `numpy`, если он установлен, используется как ускоритель матричных операций —
  но не требуется.
* Файлы моделей — обычный JSON, их можно читать из любого языка.
