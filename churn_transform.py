import tensorflow as tf
import tensorflow_transform as tft

NUMERICAL_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']
CATEGORICAL_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]
LABEL_KEY = 'Churn'

def transformed_name(key):
    return key + '_xf'

def preprocessing_fn(inputs):
    outputs = {}
    
    # 1. Normalisasi Fitur Numerik & Handle String Kosong pada TotalCharges
    for key in NUMERICAL_FEATURES:
        if key == 'TotalCharges':
            fill_missing = tf.strings.to_number(inputs[key], out_type=tf.float32)
            fill_missing = tf.where(tf.math.is_nan(fill_missing), tf.zeros_like(fill_missing), fill_missing)
            outputs[transformed_name(key)] = tft.scale_to_z_score(fill_missing)
        else:
            outputs[transformed_name(key)] = tft.scale_to_z_score(tf.cast(inputs[key], tf.float32))
            
    # 2. Transformasi Fitur Kategorikal menjadi Vocabulary Index
    for key in CATEGORICAL_FEATURES:
        if key == 'SeniorCitizen':
            str_input = tf.strings.as_string(inputs[key])
            outputs[transformed_name(key)] = tft.compute_and_apply_vocabulary(str_input)
        else:
            outputs[transformed_name(key)] = tft.compute_and_apply_vocabulary(inputs[key])
            
    # 3. Encoding Label Target (Churn: Yes/No -> 0/1)
    label_indices = tft.compute_and_apply_vocabulary(inputs[LABEL_KEY])
    outputs[transformed_name(LABEL_KEY)] = tf.cast(label_indices, tf.int64)
    
    return outputs
