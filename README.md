# Hybrid ML + Arduino Grape Sorter

This system sorts grapes based on:
1. **External Features** (Image Processing + CNN)
2. **Internal Features** (NIR spectroscopy data)
3. Learns over time to skip NIR if image model confidence is high.

---

## Steps to Use
1. **Collect Data**
   - Capture grape images → `data/images/`
   - Record NIR spectroscopy values → `data/labels.csv`

2. **Train Models**
```bash
python scripts/train_cv.py
python scripts/extract_cv_features.py
python scripts/train_hybrid.py
