# Eksperimen_SML_Naimatul-Ulumiyah

Repository berisi eksperimen dan pipeline preprocessing Sepsis ICU dataset untuk tugas SMSML.

## Struktur Proyek

```
Eksperimen_SML_Naimatul-Ulumiyah/
├── preprocessing/
│   ├── Eksperimen_Naimatul-Ulumiyah.ipynb   # notebook eksplorasi & preprocessing manual
│   ├── automate_Naimatul-Ulumiyah.py        # script otomatis preprocessing (skilled)
│   └── sepsis_preprocessing/                 # hasil preprocessing + raw files
├── Membangun_model/
│   ├── modelling.py                         # training model (MLflow autolog)
│   ├── modelling_tuning.py                  # tuning & manual MLflow logging
│   ├── requirements.txt                     # dependencies
│   └── sepsis_preprocessing/                # salinan data preprocessed untuk modelling
├── Monitoring_dan_Logging/                  # asset untuk kriteria 4
│   ├── inference.py
│   ├── prometheus_exporter.py
│   └── prometheus.yml
├── Workflow-CI/                             # placeholder folder kriteria 3
└── README.md                                # file ini
```

## Cara Memakai

1. **Preprocessing**  
   - Notebook berada di `preprocessing/Eksperimen_Naimatul-Ulumiyah.ipynb` untuk EDA & tahap manual.  
   - Jalankan `python preprocessing/automate_Naimatul-Ulumiyah.py` untuk menghasilkan `sepsis_preprocessing_train.csv` dan `test_preprocessed.csv`.

2. **Training Model**
   - Masuk ke folder `Membangun_model` (atau jalankan script dari root).  
   - Install dependensi: `pip install -r Membangun_model/requirements.txt`  
   - Jalankan `python Membangun_model/modelling.py` untuk melatih RandomForest dengan MLflow autolog.  
   - (Opsional) `python Membangun_model/modelling_tuning.py` untuk tuning lebih lanjut.

3. **MLflow UI**  
   - Setelah training, buka `http://localhost:5000` untuk melihat run dan artefak.

4. **Kriteria Lain**  
   - Folder `Monitoring_dan_Logging` berisi skrip servis & konfigurasi Prometheus.  
   - Folder `Workflow-CI` akan diisi dengan workflow GitHub Actions di kriteria berikutnya.

Semua path relatif diatur agar script dapat dijalankan langsung dari dalam folder masing-masing.
