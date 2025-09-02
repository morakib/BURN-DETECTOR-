# 🔥 Burn Classification and Triage System  

This repository contains our work on **AI-driven burn injury assessment and triage**, presented in our research poster *“Triage Made Accurate, Portable, and Instant.”*  

The system automates:  
- **Burn depth classification** (degree of burn)  
- **%TBSA (Total Body Surface Area) calculation**  
- **Fluid resuscitation recommendation** using the Parkland formula  
- **Instant triage color coding** (Red / Yellow / Green) to support quick decision-making in critical scenarios  

---

## 📊 Dataset  

We used the publicly available **Skin Burn Dataset** by [Shubham Baid on Kaggle](https://www.kaggle.com/datasets/shubhambaid/skin-burn-dataset).  
The dataset includes labeled images of different burn severities for training deep learning models.  

---

## ⚙️ Training Notebook  

Model training was done in Kaggle using this notebook:  
👉 [Skin Burn Prediction Notebook](https://www.kaggle.com/code/t3553r4ct/skin-burn-prediction)  

The pipeline included:  
1. **Data preprocessing**  
   - Image resizing, normalization, and augmentation for better generalization.  
2. **Model training**  
   - Deep learning classifier to detect burn severity.  
3. **Evaluation**  
   - Accuracy and classification performance metrics.  
4. **Integration**  
   - Automated %TBSA calculation.  
   - Parkland formula implementation for fluid resuscitation guidance.  
   - Triage system output (RED/YELLOW/GREEN).  

---

## 🚀 Features  

- **Automated burn classification** → Detects burn degree with AI.  
- **%TBSA estimation** → Calculates affected body surface area.  
- **Fluid resuscitation guide** → Recommends fluid replacement based on the Parkland formula.  
- **Instant triage system** → Reduces misclassification, delays, and mortality in emergencies.  
- **Portable & offline** → Suitable for disaster zones and rural healthcare settings.  

---

## 📌 Future Work  

- Clinical validation with real-world medical data  
- Optimized imaging fusion (RGB + thermal + NIR)  
- Enhanced affordability and integration into mobile workflows  
- Advanced privacy-preserving features  

---

## 📖 Reference  

- Dataset: [Skin Burn Dataset (Kaggle)](https://www.kaggle.com/datasets/shubhambaid/skin-burn-dataset)  
- Training Notebook: [Skin Burn Prediction (Kaggle)](https://www.kaggle.com/code/t3553r4ct/skin-burn-prediction)  
- Related Publication: [Burn Triage Study (PLOS ONE)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0206427)  

---

## ⚠️ Disclaimer  

This project is for **research and educational purposes only**.  
It is **not a substitute for professional medical diagnosis or treatment**.  
As a small amount of dataset, Model didn't learn well, It will be better for large amount of dataset.
Always consult a licensed medical professional for clinical decisions.  
