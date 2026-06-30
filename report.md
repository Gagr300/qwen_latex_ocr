## Выбор метрики

### edit_distance
**Edit Distance (Расстояние Левенштейна)** - это минимальное количество операций вставки, удаления и замены, необходимых для превращения одной строки в другую.

### BLEU-4
**BLEU-4 (Bilingual Evaluation Understudy, 4-gram)** - метрика для оценки генерации текста.  

Подходит, поскольку нам не требуется оценить смысл слов (не нужны синонимы). Нужно искать точные совпадения подпоследовательностей.

$$BLEU = BP \times \exp(\sum_{n=1}^Nw_n\ln{P_n})$$
$BP$ — штраф за краткость для штрафования коротких ответов, $P_n$ — точность n-грамм, а $w_n$ — веса для каждого уровня $n$-грамм. 

### TeXBLEU
**TeXBLEU** используется для измерения сходства между текстами, содержащими математические выражения.

_Как считается:_

1. Предобработка. Чтобы обеспечить единообразие интервалов.

2. Токенизация и эмбеддинг. Используется собственный токенизатор BPE (кодировка пар байтов), созданный с помощью корпуса 172 000 статей с arXiv.

3. Вычисляется расстояние между токенами:
$$d(t1, t2) = (cosDist(e1, e2)^\alpha + \frac{tanh(\beta · |p1 - p2|))}{2}$$
$$cosDist(e1, e2) = 1 - cosSim(e1, e2)$$

4. Вычисляется сходство n-грамм:
$$sim_n(R, P) = 1 - \frac{\sum_{i=1}^{L_n} \sum_{j=1}^n d(r_{ij}, p_{ij})}{L_n \cdot n}$$
$n$ - длина n-грамм, $L_n$ - количество n-грамм

5. Вычисляется итоговая метрика:
$$TeXBLEU = exp(\sum_{n=1}^N w_n \log sim_n(R, P))$$


### Exact Match
**Exact Match (точное совпадение)** - доля точных совпадений.

### perplexity
**Perplexity** показывает, насколько модель "удивлена" тестовым данным.


## Модель

### Архитектура

- Vision Encoder:
  - 24-слойный ViT 
  - размерность скрытого состояния 1024 
  - 16 голов внимания
  - патчи размером 16x16
  - Deep Stack: объединяет информацию из нескольких слоев кодировщика ViT: 8, 16, 24
  - Interleaved-MRoPE: MRoPE c явными метками времени (timestamp tokens) и (THWTHW...) (вместо (TT...HH...WW))

- Qwen3 LM Dense: 
  - размерность скрытого состояния - 2048
  - количество слоев - 28
  - 16 голов внимания с Grouped-Query Attention (GQA) (8 KV-голов)
   /*
    Multi-Head Attention (MHA) - для каждого Query модель вычисляет отдельную пару Key-Value для каждой Head. 
    Multi-Query Attention (MQA) - для каждого Query модель вычисляет одна пара Key-Value для каждой Head.
    Grouped-Query Attention - делит все головы внимания на группы.
    16 голов разбиты на 8 групп (по 2 головы в каждой).
   */
  - в MLP-блоках функция активации - SiLU (SwiGLU)
   SiLU $SiLU(x)=x⋅\sigma(x)$, где $\sigma(x)$ - это стандартная сигмоида $\frac{1}{1+e^{−x}}$.
  - поддерживает контекстное окно до 262,144 токенов

### Особенности

- OCR: поддерживает 32 языка, хорошо работает при слабом освещении, размытии и наклоне, с редкими символами 

## Подбор параметров обучения

Первая попытка была выполнена с параметрами, взятыми из ноутбука "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
```python
per_device_train_batch_size = 2,
gradient_accumulation_steps = 4,
warmup_steps = 5,
num_train_epochs = 1, 
learning_rate = 2e-4,
logging_steps = 1,
optim = "adamw_8bit",
weight_decay = 0.001,
lr_scheduler_type = "linear",

remove_unused_columns = False,
dataset_text_field = "latex",
dataset_kwargs = {"skip_prepare_dataset": True},
max_length = 2048,
```


```python
OPTIM = "adamw_8bit"
LR_SCHEDULER_TYPE = "cosine" 

EVAL_STRATEGY = "steps"
SAVE_STRATEGY = "steps"
SAVE_TOTAL_LIMIT = 2 

WEIGHT_DECAY = 0.001
WARMUP_STEPS = 4

LORA_DROPOUT = 0 

REMOVE_UNUSED_COLUMNS = False
DATASET_KWARGS = {"skip_prepare_dataset": True} 

METRIC_FOR_BEST_MODEL="eval_loss"
LOAD_BEST_MODEL_AT_END = True
```


Параметры, которые показали наилучшие результаты при обучении на датасете linxy: 

```python
LEARNING_RATE = 2e-4
LORA_RANK = 16
LORA_ALPHA = 16
EPOCHS = 2
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
``` 

Результаты: 
 - edit_distance: 0.00580     
 - bleu4:         0.98201
 - texbleu:       0.99453 
 - exact_match:   0.9285  

При этом были следующие ошибки:


Параметры, которые показали наилучшие результаты при обучении на датасете linxy + deepcopy: 

```python
LEARNING_RATE = 2e-5
LORA_RANK = 16
LORA_ALPHA = 16
EPOCHS = 2
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
```
**Результаты:** 
 - edit_distance: 0.00448 
 - bleu4:         0.98665
 - texbleu:       0.99621
 - exact_match:   0.94286  

При этом были следующие ошибки:
- не были добавлены фигурные скобки (которые на самом деле не очень нужны в тех примерах)
- не было подписано \left и \right перед круглыми скобками (тоже не особо нужны в тех примерах) 



## Пара моментов:
- слишком маленькие выборки test и val, содержат данные только из первого датасета
- разные типы ответов (из-за пробелов)
- вероятно начали подстраиваться под более "корявые почерки", например, в '|' начали видеть '1', а в 'D' - 'P'
- необходима предобработка изображений или нужны грязные данные при обучении (в собственном примере, можно заметить, что на одном и том же изображении, но с более хорошей обработкой модели в среднем лучше работают)