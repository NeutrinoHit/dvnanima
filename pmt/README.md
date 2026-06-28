# PMT Static Scene Generator

Конфигурируемый расчёт 2D-траекторий электронов в упрощённой модели ФЭУ с сохранением статических сцен (`PNG`) и опционального набора данных (`HDF5`).

## Структура

- `main.py` — CLI запуск одной сцены или пакета сцен.
- `physics/` — геометрия, поле, интегратор траекторий, оркестрация расчёта.
- `render/` — статический рендер сцены в PNG (matplotlib).
- `io/` — HDF5 и генерация выходных путей.
- `config/` — `base.toml` + `scenes/*.toml`.
- `tests/` — unit tests.

## CLI

Одна сцена:

```bash
cd pmt
python main.py --base config/base.toml --scene config/scenes/surface_to_plate.toml --output out/surface_to_plate.png
```

Пакетный запуск всех сцен:

```bash
cd pmt
python main.py --base config/base.toml --scene-dir config/scenes --output-dir out
```

Опциональный HDF5:

```bash
cd pmt
python main.py --base config/base.toml --scene config/scenes/point_field.toml --output out/point_field.png --hdf5 out/point_field.h5
```

`dry-run` (без расчёта, печать merged-конфига):

```bash
cd pmt
python main.py --base config/base.toml --scene config/scenes/point_field.toml --dry-run
```

## Поддерживаемые сцены

- `point_field`
- `surface_to_point`
- `surface_to_plate`
- `surface_to_plate_voltage_scan`
- `surface_to_plate_with_focus`

## Режимы физики

- `scene.physics_mode = "central"` — аналитическое центральное поле.
- `scene.physics_mode = "numerical"` — численное поле на сетке через решение уравнения Лапласа.

Траектории во всех режимах интегрируются единым `E + v × B` интегратором; `Bz=0` по умолчанию.

## Скрипты

- `50_run_single_scene.sh` — запуск одной сцены.
- `51_run_all_scenes.sh` — пакетный запуск всех сцен.
- `52_dry_run_scene.sh` — dry-run одной сцены.
- `20_run_tests.sh` — запуск тестов.

## Тесты

```bash
cd pmt
./20_run_tests.sh
```

