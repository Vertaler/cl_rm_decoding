# Описание алгоритма
В данном репозитории приведена параллельная реаилизация  
декодирования кода Рида-Маллера. Для нелинейных функций  
применяется мажоритарный алгоритм Рида. А для линейных
алгоритм Лицына-Шеховцева.
Более подробное описание приведено в документе [RM_Notes.pdf](https://github.com/Vertaler/cl_rm_decoding/blob/master/RM_notes.pdf)

# Инструкция по использованию

## 1) установка зависимостей
* [Miniconda c Python 3.7](https://conda.io/miniconda.html)
* [pyopencl](https://documen.tician.de/pyopencl/misc.html)  

## 2) запуск
Для запуска необходимо выбрать интерпретатор языка python3,  
который поставляется вместе с conda.

### 2.1
Запуск интерактивного декодирования 
```bash
python interactive_decoding.py
```
### 2.2
Запуск скрипта сравнения производительности:
```bash
python run_benchmark.py число_итераций степень_функции порядок_рм_кода 
```
Или просто
```bash
python run_benchmark.py 
```
Это эквивалентно:
```bash
python run_benchmark.py 1000 9 2
```

При этом параметр степень_функции должен быть меньше 10, так как 
иначе последовательный алгоритм зависает из-за какой-то ошибки в коде(

