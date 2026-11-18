<div align="center">

<img src="https://n.uguu.se/hHPnmlZBhqJw.png" alt="Lume Studio" width="880">

# Lume

**Простой язык программирования для создания ИИ-моделей + оболочка Lume Studio с ИИ-ассистентом.**

*A tiny programming language for building AI models, shipped with Lume Studio — an IDE with a built-in keyless AI assistant.*

</div>

<p align="center">
  <a href="docs/screenshots/index.html">📸 все скриншоты</a> ·
  <a href="docs/TUTORIAL.md">📖 руководство</a> ·
  <a href="docs/LANGUAGE.md">📘 спецификация</a> ·
  <a href="examples/">🧪 примеры</a>
</p>

---

## Что это такое / What is this

| | |
| --- | --- |
| 🗣 **Язык** | `Lume` — 30 ключевых слов, фигурные скобки, знакомый синтаксис. Читается как обычный код, но модели описываются в три строки. |
| 🧠 **Модели** | Слои `Dense`, `Activation`, `Dropout`, `Flatten`, `Embedding`, оптимизаторы `sgd/momentum/rmsprop/adam`, лоссы `mse/mae/binary_crossentropy/crossentropy`, метрики и матрица ошибок — из коробки. |
| 🖥 **Оболочка** | `Lume Studio` — IDE с подсветкой синтаксиса, консолью, деревом файлов и встроенным **ИИ-ассистентом Pollinations без API-ключа**. |
| 📦 **Сборка** | Из одного и того же кода собираются: **модель** (`model.json` + загрузчик), **приложение** (готовый веб-сервис) и **EXE** (одиночный исполняемый файл). |
| ⚡ **Зависимости** | Ноль. Только стандартная библиотека Python 3.9+. |

---

## Быстрый старт / Quick start

```bash
git clone https://github.com/Hantikos/lume.git
cd lume

# 1. запустить пример
python -m lume run examples/xor.lume

# 2. открыть оболочку (откроется окно с редактором)
python -m lumeide

# 3. создать свой проект
python -m lume new my-project
```

Установить как команды в системе:

```bash
pip install -e .
lume run examples/xor.lume
lumeide
```

---

## Язык за 60 секунд / The language in 60 seconds

```lume
# комментарий
print "Hello, Lume!"

fn square(x) {
    return x * x
}

let name = "world"
print f"привет, {name}: {square(7)}"

# модель - это просто блок
data points = dataset.xor()

model XorNet {
    layer Dense(16, "relu")
    layer Dense(1, "sigmoid")

    train(data = points, epochs = 150, lr = 0.2, optimizer = "adam", loss = "binary_crossentropy")
    eval(data = points)
}

print predict(XorNet, [[1, 0]])    # [1]
```

### Возможности языка

* Переменные: `let`, `const`, присваивание, присваивание по индексу `items[0] = 5`
* Функции: аргументы, значения по умолчанию, рекурсия, замыкания
* Типы: `int`, `float`, `str`, `bool`, `null`, `list`, `dict`, `tensor`, `model`, `dataset`
* Ветвления и циклы: `if / else`, `for x in list`, `while`, `break`, `continue`
* Строки: `f"интерполяция {x}"`, `split`, `join`, `upper`, `replace`, ...
* Тензоры: `tensor([[1,2],[3,4]])`, оператор матричного умножения `@`
* Модули: `import math`, `import json`
* Модели: `model`, `layer`, `train`, `eval`, `predict`, `forward`, `save`, `load`

Полная спецификация: **[docs/LANGUAGE.md](docs/LANGUAGE.md)** · Пошаговое руководство: **[docs/TUTORIAL.md](docs/TUTORIAL.md)**

---

## Lume Studio

<img src="https://d.uguu.se/cJyMMyjhNEAY.png" alt="Lume Studio — запуск" width="880">

* **Редактор** с подсветкой синтаксиса, номерами строк и подсветкой строки с ошибкой
* **Консоль** вывода: логи обучения, метрики, ошибки; вкладки «Проблемы» и «Модели»
* **Дерево файлов** и список готовых примеров
* **ИИ-ассистент** — Pollinations, работает **без API-ключа**:
  объясняет код, находит ошибки, пишет модели, отвечает на вопросы по синтаксису
* **Генерация картинок** по описанию (для обложек, схем, иллюстраций)
* **Сборка** в один клик: модель → приложение → EXE
* Тёмная и светлая темы, горячие клавиши: `Ctrl+Enter` — запуск, `Ctrl+S` — сохранить

Запуск:

```bash
python -m lumeide                    # обычный запуск
python -m lumeide --workspace ~/projects
python -m lumeide --port 8080        # свой порт
python -m lumeide --no-window        # без окна (для CI/сервера)
```

---

## Сборка / Build

### Модель

```bash
python -m lume build model examples/xor.lume -o build
# build/XorNet/model.json   — веса и архитектура
# build/XorNet/model.py     — загрузчик на чистом Python
```

### Приложение (готовый веб-сервис с интерфейсом)

```bash
python -m lume build app examples/xor.lume -o build
# build/XorNetApp/app.py        — сервер + вывод предсказаний
# build/XorNetApp/index.html    — интерфейс
# build/XorNetApp.zip           — всё вместе
python build/XorNetApp/app.py   # http://127.0.0.1:8000
```

### EXE (одиночный исполняемый файл)

```bash
pip install pyinstaller
python tools/build_exe.py --all      # dist/lume(.exe) и dist/LumeStudio(.exe)
```

Готовые `.exe` для Windows собираются в CI: **[workflows/build.yml](workflows/build.yml)** —
скопируйте его в `.github/workflows/` (см. [workflows/README.md](workflows/README.md)) и запустите
workflow на GitHub Actions, затем заберите артефакт.

---

## CLI

```
lume run <file.lume>        запустить программу
lume check <file.lume>      проверить синтаксис
lume repl                   интерактивный режим
lume new <project>          создать проект
lume build model <file>     собрать модель
lume build app <file>       собрать приложение
lume examples               список примеров
```

---

## Примеры

| файл | что показывает |
| --- | --- |
| [`examples/hello.lume`](examples/hello.lume) | функции и печать |
| [`examples/tour.lume`](examples/tour.lume) | полный тур по языку |
| [`examples/xor.lume`](examples/xor.lume) | нейросеть на 100 % решает XOR |
| [`examples/regression.lume`](examples/regression.lume) | линейная регрессия: учим `y = 3x + 2` |
| [`examples/text_sentiment.lume`](examples/text_sentiment.lume) | классификатор тональности текста на bag-of-words |

---

## Структура проекта

```
lume/            интерпретатор языка (лексер, парсер, рантайм, нейросети)
lumeide/         оболочка: локальный сервер + веб-интерфейс студии
examples/        примеры на Lume
tests/           тесты языка, сборщиков и API (python -m unittest discover -s tests)
tools/           сборка EXE, скриншоты, публикация на GitHub
workflows/       готовые GitHub Actions workflows (скопируйте в .github/workflows)
docs/            спецификация языка, руководство, скриншоты
```

> Скриншоты хранятся в репозитории дважды: как `docs/screenshots/*.png.b64`
> (файловый API GitHub принимает только текст) и как самодостаточная страница
> `docs/screenshots/index.html` со встроенными картинками. Восстановить обычные
> PNG-файлы: `python tools/decode_screenshots.py`.

## Тесты

```bash
python -m unittest discover -s tests -v
```

## Дорожная карта

* свёрточные слои (`Conv2d`, `MaxPool`) и операции над изображениями
* рекуррентные слои (`LSTM`, `GRU`) и работа с последовательностями
* импорт и экспорт в ONNX
* плагины и пакеты `lume` с реестром

## Лицензия

MIT — см. [LICENSE](LICENSE).

---

<div align="center">
  <sub>Сделано с ❤️ для тех, кто хочет писать нейросети без лишнего шума.</sub>
</div>
