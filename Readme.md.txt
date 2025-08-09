Brain Tumor Segmentation Web Application
This repository contains a web application for automatic brain tumor segmentation from MRI slices, inspired by the research paper:

Bridged-U-Net-ASPP-EVO and Deep Learning Optimization for Brain Tumor Segmentation by Rammah Yousef et al., 2023.

The app enables users to upload an MRI image and receive a segmented output highlighting tumor regions in red.

🚀 Features
Upload & Process brain MRI images (PNG or JPEG)

Deep Learning Segmentation powered by a U-Net-based architecture with Atrous Spatial Pyramid Pooling (ASPP) and squeeze–excitation blocks

Two-step interface: Upload page → Segmentation result page

Example Images included for demonstration without requiring the trained weights

FastAPI backend for inference

Responsive HTML/CSS frontend

📂 Project Structure
bash
Copy
Edit
brain_tumor_webapp/
│
├── app.py                   # FastAPI server
├── model.py                 # Bridged-U-Net-ASPP-EVO model definition
├── requirements.txt         # Dependencies
├── weights/                 # Place trained model weights here (weights.pt)
│
├── templates/
│   └── index.html            # Main upload & result page
│
├── static/
│   ├── css/style.css         # Styling
│   └── images/               # Example MRI + segmentation images
│
└── preview_*.html            # Static mockups for demonstration screenshots
🖥️ How It Works
Upload an MRI brain slice (PNG/JPG).

The app preprocesses the image.

If trained weights are available, the U-Net model performs segmentation. Otherwise, a fallback mask is generated.

The result is displayed with the tumor area highlighted in red.

📸 Screenshots
Upload Page

Result Page

🔧 Installation & Usage
Clone the repository

bash
Copy
Edit
git clone https://github.com/yourusername/brain_tumor_segmentation_webapp.git
cd brain_tumor_segmentation_webapp
Create a virtual environment (optional but recommended)

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Add trained weights (optional)
Place your weights.pt file in the weights/ folder.

Run the app

bash
Copy
Edit
uvicorn app:app --reload
Open in browser
Go to http://127.0.0.1:8000

📄 Requirements
Python 3.8+

FastAPI

Uvicorn

Pillow

torch (if using the trained model)

python-multipart

📚 Reference
If you use this project in your research or development, please cite:

R. Yousef, Bridged-U-Net-ASPP-EVO and Deep Learning Optimization for Brain Tumor Segmentation, Diagnostics, 2023.