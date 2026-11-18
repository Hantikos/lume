# CI / CD для Lume

Эти файлы — готовые GitHub Actions workflows. GitHub читает workflows только из
каталога `.github/workflows/`, поэтому скопируйте их туда:

```bash
cp workflows/build.yml workflows/screenshots.yml .github/workflows/
git add .github/workflows
git commit -m "Enable CI"
git push
```

Или сделайте это прямо в интерфейсе GitHub: **Add file → Create new file →
`.github/workflows/build.yml`** и вставьте содержимое файла.

## build.yml

Собирает `lume` и `LumeStudio` в однофайловые исполняемые файлы под
Windows, Linux и macOS, прогоняет smoke-тест и выкладывает артефакты.
По тегу `v*` дополнительно публикует релиз.

Артефакты можно забрать на странице репозитория → **Actions** → **Artifacts**.

## screenshots.yml

Восстанавливает PNG-скриншоты из их base64-копий (`docs/screenshots/*.png.b64`)
и удаляет служебные файлы. Нужен потому, что файловый API GitHub принимает
только текст.
