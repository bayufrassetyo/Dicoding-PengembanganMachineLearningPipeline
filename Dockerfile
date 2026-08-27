# Menggunakan image TensorFlow Serving resmi yang stabil
FROM tensorflow/serving:latest

# Menyalin folder model lokal kita ke dalam direktori model internal kontainer Docker
COPY ./serving_model_dir /models/customer_churn_model

# Mengatur variabel lingkungan (environment variable) untuk nama model
ENV MODEL_NAME=customer_churn_model
