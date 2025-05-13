#**SAR Image Colorization using Deep Learning**


**🛰️ Project Overview**      
        This project is focused on building an intelligent system that utilizes Synthetic Aperture Radar (SAR) satellite imagery for accurate crop monitoring and classification. Unlike optical satellite data, SAR can capture information regardless of weather conditions or lighting, making it especially valuable for agricultural surveillance in monsoon-prone or cloud-covered regions. The system leverages machine learning and deep learning techniques to classify various crop types based on their radar signal signatures. By pre-processing SAR data, extracting relevant features, and training predictive models, the application is able to analyze agricultural areas and identify crop types with high accuracy. The aim is to assist farmers, agronomists, and policy-makers in making informed decisions about crop health, yield forecasting, and land-use planning. By enabling early detection of crop patterns and changes, the project enhances food security and promotes sustainable agricultural practices. This project was developed as part of a college group initiative under the mentorship of ISRO, contributing to national-level goals in remote sensing and smart agriculture.To address this, we implemented a U-Net-based Convolutional Neural Network (CNN) model to generate colorized outputs from grayscale SAR images, enhancing visual clarity and usability. This project was conducted under the guidance of ISRO (Indian Space Research Organisation).

**🎓 Academic Details**
      **Project Title:** SAR Image Colorization using Deep Learning
      **Project Type:** Final Year College Group Project
      **University:**  PRESIDENCYUNIVERSITY , BENGALURU
      **Department:** Computer Engineering (Artificial Intelligence & Machine learning)
      **Academic Year:** 2024–2025

**Team Leader**
    GODAVARTHI NAGA MANOJ BALAJI (Me)
**Team Members:**
    1. GOLLA MEGHANA
    2. MUNAGALA NAGA SAMEERA
    3.  MUNNANGI SUMA REDDY
    4.  AYUSH CHHETRI


**Project Guide:** Dr.SMITHA PATIL ,Assistant Professor(Selection Grade) 


**🎯 Objective**
      To develop a deep learning-based system capable of transforming grayscale SAR images into colorized RGB versions, enabling better interpretation and support for land and environmental analysis.



**🧠 Technologies Used**
      1. Python
      2. TensorFlow / Keras
      3. OpenCV
      4. NumPy
      5. Matplotlib

**📂 Project Structure**

        ├── dataset/                # Contains SAR grayscale and reference RGB images
        ├── model/                  # Saved trained models
        ├── notebooks/              # Jupyter notebooks for development
        ├── src/                    # Core Python scripts (data loading, training, inference)
        ├── outputs/                # Colorized SAR image outputs
        ├── Final Report.pdf        # Full project report
        ├── README.md               # Project description and info


**🌱 Sustainable Development Goal (SDG) Contribution**
            Goal 15: Life on Land
            This project supports SDG Goal 15 by enhancing remote sensing imagery, enabling more effective monitoring of forests, vegetation, land use patterns, and biodiversity. This aids efforts in land conservation, habitat protection, and sustainable ecosystem management.

**🛠️ Setup and Installation**
      Follow the steps below to set up and run the project in Visual Studio Code:
      
      **1️⃣ Open the Project**
            1. Launch Visual Studio Code
            2. Open the integrated terminal
      
       .
      **2️⃣ Set Up Virtual Environment**
              1. In the terminal, select Bash
              2. Create a virtual environment:  python -m venv .venv
              3. Activate it (based on shell):
                1. Bash (Linux/macOS):  source .venv/bin/activate
                2. PowerShell (Windows): .venv\Scripts\Activate.ps1
                3. Command Prompt (cmd.exe): .venv\Scripts\activate.bat
              
      **3️⃣ Install Dependencies**  Once the environment is activated, install all required libraries:  pip install -r requirements.txt
      **4️⃣ Run the Application**  Execute the main program using:   python mainv2.pyw
      

**📦 Requirements**
      The following Python packages are required for this project:
      
      1. numpy (version below 2.0)
      2. Pillow
      3. tensorflow
      4. keras
      5. scikit-image
      6. opencv-python
      7.  matplotlib
      
      All dependencies are listed in the requirements.txt file. Install them using the pip install -r command as described above.

**Output:**
     This research demonstrates the successful application of deep learning, particularly a dual-stream Convolutional Neural Network (CNN), to colorize Synthetic Aperture Radar (SAR) images, transforming them from grayscale to more intuitive color representations. By integrating advanced image processing with an accessible user interface, we have improved the interpretability of SAR imagery, expanding its use in sectors like environmental monitoring, urban planning, and disaster management. Our results show significant improvements in both quantitative metrics (PSNR, SSIM) and qualitative assessments, with domain experts noting enhanced detail recognition and usability. Custom metrics for evaluating color accuracy for different terrain types further highlight the practical applications of this work.
    The methodology, which combined transfer learning, extensive data augmentation, and user feedback, successfully addressed challenges like the limited availability of labeled SAR data and speckle noise. The interactive GUI not only facilitated user engagement but also enabled continuous refinement of the model based on real-world feedback. This research proves that colorized SAR images can enhance decision-making processes by more accurately depicting land cover and improving image usability for various applications.



**📑 License**
      This project is for educational and research use only. For other uses or collaboration inquiries,please contact the authors.




For other uses or collaboration inquiries,and for any information please contact the authors.
@GODAVARTHI NAGA MANOJ BALAJI 
godaverthinagamanojbalaji@gmail.com 
