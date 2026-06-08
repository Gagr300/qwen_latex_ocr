# Task 1 Fine-tuning VLM

**Task:** Fine-tune any Vision-Language Model for converting an input image containing a handwritten formula into LaTeX format.


## Experimental setup & Training hyperparameters


> [!abstract] Setup
> **GPU:** T4 x2
> **Python:** 3.12.13
> **torch:** 2.10.0+cu128
> **unsloth:** 2026.6.1



- **Chosen model:** `Qwen/Qwen3-VL-2B-Instruct`
![[Pasted image 20260606001024.png|626]]

- **Datasets:**

| dataset                             | columns          | train   | val    | test  |
| ----------------------------------- | ---------------- | ------- | ------ | ----- |
| `linxy/LaTeX_OCR (human_handwrite)` | 'image', 'text'  | 1 200   | 68     | 70    |
| `deepcopy/MathWriting-Human`        | 'image', 'latex' | 229 864 | 15 674 | 7 644 |

There is a difference in latex:
- there are much more spaces in `linxy/LaTeX_OCR` than in `deepcopy/MathWriting-Human`.

## Comparison


> [!cite] 
> In [research](https://github.com/LinXueyuanStdio/LaTeX_OCR), the linxy/LaTeX_OCR dataset was created for, were used four metrics: perplexity, EditDistance, BLEU-4 and ExactMatchScore.

|                                                              | perplexity | EditDistance | BLEU-4  | TeXBLEU | ExactMatchScore | ```sparse_softmax_cross_entropy_with_logits``` |
| ------------------------------------------------------------ | ---------- | ------------ | ------- | ------- | --------------- | ---------------------------------------------- |
| **Zero-shot inference**                                      |            | 0.55326      | 0.07669 | 0.75931 | 0.0             |                                                |
| **One-shot inference**                                       |            | 0.32362      | 0.33414 | 0.78631 | 0.01429         |                                                |
| **Two-shot inference**                                       |            | 0.29282      | 0.39222 | 0.83945 | 0.22857         |                                                |
| **SFT (linxy/LaTeX_OCR:train)**                              |            | 0.00580      | 0.98201 | 0.99453 | 0.9285          |                                                |
| **SFT (linxy/LaTeX_OCR:train + deepcopy/MathWriting-human)** |            | 0.00448      | 0.98665 | 0.99621 | 0.94286         |                                                |




> A good idea is proposed by Harvard’s paper. First, slice the image vertically. Then, compare the edit distance between these slices [link]([Seq2Seq for LaTeX generation](https://guillaumegenthial.github.io/image-to-latex.html))

**Some real examples:**

| image                                     | real                                                       | Zero-shot                                                       | **One-shot**                                                                 | Two-shot                                                          | **SFT (LaTeX_OCR)**                                                 | **SFT (LaTeX_OCR + MathWriting-human)**                                |
| ----------------------------------------- | ---------------------------------------------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| ![[Pasted image 20260603152842.png\|300]] | `\\sqrt { b ^ { 2 } - 4 a c }`                             | ` ```latex\n\\sqrt{b^2 - 4ac}\n``` `                            | ` ```latex\n\\sqrt{b^2 - 4ac}\n``` `                                         | ` \\sqrt { b ^ { 2 } - 4 a c } `                                  | ` \\sqrt { b ^ { 2 } - 4 a c } `                                    | ` \\sqrt{b^{2}-4ac} `                                                  |
| ![[Pasted image 20260603152849.png]]      | `\\sqrt { x - y - z + x ^ { 2 } + y ^ { 2 } + z ^ { 2 } }` | ` ```latex\n\\sqrt{x - y - z + x^2 + y^2 + z^2}\n``` `          | ` ```latex\n\\sqrt{x - y - z + x^2 + y^2 + z^2}\n``` `                       | ` \\sqrt{x - y - z + x^2 + y^2 + z^2} `                           | ` \\sqrt { x - y - z + x ^ { 2 } + y ^ { 2 } + z ^ { 2 } } `        | ` \\sqrt { x - y - z + x ^ { 2 } + y ^ { 2 } + z ^ { 2 } } `           |
| ![[Pasted image 20260603153247.png]]      | `\\frac { 2 \\tan \\alpha } { 1 - \\tan ^ { 2 } \\alpha }` | ` ```latex\n\\frac{2 \\tan \\alpha}{1 - \\tan^2 \\alpha}\n``` ` | ` $$\\frac { 2 \\tan \\alpha } { 1 - \\tan ^ { 2 } \\alpha }$$ `             | ` \\frac { 2 \\tan \\alpha } { 1 - \\tan^2 \\alpha } `            | ` \\frac { 2 \\tan \\alpha } { 1 - \\tan ^ { 2 } \\alpha } `        | ` \\frac { 2 \\tan \\alpha } { 1 - \\tan ^ { 2 } \\alpha } `           |
| ![[photo1 1.jpg]]                         |                                                            | ` ```latex \sqrt{x - y + z - x^2 + y^2 + z^3} = 100 ``` `       | ` ```latex \sqrt{x - y + z - x^2 + y^2 + z^3} = 100 ``` `                    | ` \sqrt { x - y + z - x ^ { 2 } + y ^ { 2 } + z ^ { 3 } } = 100 ` | ` \sqrt { x - y + z - x ^ { 2 } + y ^ { 2 } + z ^ { 3 } } = 1 0 0 ` | ` \sqrt { x - y + z - x ^ { 2 } + y ^ { 2 } + z ^ { 3 } } = 1 0 0<br>` |
| ![[photo2.jpg]]                           |                                                            | ` ```latex x + y + z \approx 8 ``` `                            | ` ```latex x + y + z \approx 8 ``` `                                         | ` $$x + y + z \approx 8$$ `                                       | ` x + y + z \approx 8 `                                             | ` x + y + z \approx 8 `                                                |
| ![[photo3.jpg]]                           |                                                            | ` ```latex D_{KL}(x\|\|y) ``` `                                 | ` ```latex \mathrm{P}_{\mathrm{K}} \left( x \, \mathrm{I} \, y \right) ``` ` | ` \mathrm{P}_{\mathrm{K}}\left(x \, \mathrm{I} \, y\right) `      | ` P _ { K \rightarrow } ( X 1 1 Y ) `                               | ` P _ { K \cup L } ( X \| Y ) `                                        |

Some notices:
- In zero-shot model generates text using ` ```latex `, ` ``` ` и ` $$ `, mainly with  ` ```latex ` and ` ``` `. In one-shot answers still there is the same problem, but it mainly uses `$$`. In two-shot answers still there are `$$` in approximately 50%.
- Using example in few-shot from `deepcopy/MathWriting-human` causes lower metrics due to lack of spaces. (in the table results were obtained by using example from `linxy/LaTeX_OCR:validation`)


**Loss curve for the first sft**
![[Pasted image 20260608010730.png]]
## the first sft wrong images
![[Pasted image 20260608013424.png]]
**Loss curve for the second sft**
![[Pasted image 20260608003344.png]]
# Task 2 Streamlit Application

**Task:** Develop a Streamlit application that receives an image of a handwritten formula and returns the rendered LaTeX output. 

## without photo
![[Pasted image 20260608010153.png]]
## with a photo
![[Pasted image 20260608010249.png]]
## with answer

![[Pasted image 20260608010415.png]]

![[Pasted image 20260608010609.png]]