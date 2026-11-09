from setuptools import setup, find_packages

setup(
    name="signlang-ai",
    version="2.0.0",
    author="Karthikeya Parvathareddy",
    author_email="karthikeyaparvathareddy@gmail.com",
    description="Real-Time Sign Language Translator (Video to Text + Speech) using MediaPipe & Deep Sequence Networks",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/karthikeya-parvathareddy/SignLang-AI",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "opencv-python",
        "mediapipe",
        "tensorflow",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "pyyaml",
        "pyttsx3",
        "streamlit"
    ],
    entry_points={
        "console_scripts": [
            "signlang-live=src.realtime_detector:run_realtime_app",
            "signlang-train=src.train:train",
            "signlang-evaluate=src.evaluate:run_evaluation",
        ],
    },
)
