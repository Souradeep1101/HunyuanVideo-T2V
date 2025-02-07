# HunyuanVideo-T2V

HunyuanVideo-T2V is an implementation of the *HunYuanVideo* research paper using PyTorch. The project focuses on building a scalable pipeline for **Text-to-Video (T2V)** generation using diffusion models, transformers, and multimodal language models.

## 📂 Project Structure
```
HunyuanVideo-T2V/
├── data/                     # Raw and processed datasets
├── models/                   # Model architecture files
├── scripts/                  # Data processing and training scripts
├── configs/                  # Configuration files (model, data, training)
├── utils/                    # Helper functions (e.g., video processing, embeddings)
├── logs/                     # Training and processing logs
├── checkpoints/              # Model checkpoints
├── outputs/                  # Generated videos and results
├── notebooks/                # Jupyter notebooks for experimentation
├── environment.yml           # Conda environment dependencies
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── main.py                   # Entry point (if required later)
└── LICENSE                   # MIT License
```

## 🛠 Features
- **Video Preprocessing**: Automatic splitting, frame extraction, and deduplication.
- **Caption Generation**: Structured captions using BLIP-2 or LLaVA.
- **Model Training**: Progressive learning from images to videos.
- **Text-to-Video Generation**: Uses diffusion models for high-quality video generation.

## 🔧 Setup

### 1. Clone the repository
```bash
git clone https://github.com/Souradeep1101/HunyuanVideo-T2V.git
cd HunyuanVideo-T2V
```

### 2. Set up the environment
#### Using Conda:
```bash
conda env create -f environment.yml
conda activate hunyuanvideo-t2v
```
#### Or using pip:
```bash
pip install -r requirements.txt
```

### 3. Run the scripts
- **Preprocess videos**:
  ```bash
  python scripts/preprocess_videos.py
  ```
- **Generate captions**:
  ```bash
  python scripts/generate_captions.py
  ```

## 📊 Results
Add example outputs or showcase generated videos here.

## 📜 License
This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/<your-username>/HunyuanVideo-T2V/issues).

## 📧 Contact
For any inquiries or feedback, please reach out to **rishibanerjee1101@gmail.com**.

---

## ⭐ Acknowledgments
This project is inspired by the *HunYuanVideo* research paper and builds on state-of-the-art technologies like PyTorch, BLIP-2, and diffusion models.
