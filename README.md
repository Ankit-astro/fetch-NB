# FETCH-NB (Modified Version)

This repository is a **modified fork** of the original [FETCH](https://github.com/devanshkv/fetch) project.  
It was developed as part of my **MSc project** at the Central University of Haryana, titled  
**"Enhancing narrowband FRB search sensitivity using Machine Learning."**

---

## 🔄 Modifications
- Added support for **narrowband FRB search**.
- Integrated **custom feature extraction** for narrowband pulse profiles.
- Trained models for better classification of narrowband pulses in data.
- 

---

## 🧠 Trained Models
Trained models can be downloaded from [Hugging Face](https://huggingface.co/Ankit-astro/fetch-NB).


| Model | Training Layers (FT, DMT, Fusion) | Validation Accuracy |
|:------|:--------------------------------:|:------------|
| model_Z | (1, 1, 2) | 95.77% |
| model_Z_alpha | (2, 2, 2) | 96.30% |
| model_Z_beta | (3, 2, 2) | 96.81% |

In our testing model_Z performed batter then the other two in RFI detection. 
---

## ⚙️ Usage
Same as FETCH, But now trained for classifying Narrowband bursts also more effectively .

```bash
predict.py --data_dir /data/candidates/ --model /path/to/trained/model.keras
