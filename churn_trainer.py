import os
import tensorflow as tf
import tensorflow_transform as tft
from keras_tuner.tuners import RandomSearch
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.tuner.component import TunerFnResult

# Definisi kunci fitur yang sama dengan modul transform
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

def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Fungsi pembungkus untuk parsing data mentah saat model di-deploy nanti"""
    model.tft_layer = tf_transform_output.transform_features_layer()
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')])
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)
        
    return serve_tf_examples_fn

def _input_fn(file_pattern, tf_transform_output, batch_size=64):
    """Membaca data TFRecord hasil transformasi komponen Transform"""
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=lambda filenames: tf.data.TFRecordDataset(filenames, compression_type='GZIP'),
        label_key=transformed_name(LABEL_KEY)
    )
    return dataset

def _build_model(hp, tf_transform_output):
    """Membangun arsitektur model DNN dengan Hyperparameter Tuning Keras Tuner"""
    input_layers = []
    feature_columns = []
    
    # Memetakan spesifikasi fitur hasil transformasi
    feature_spec = tf_transform_output.transformed_feature_spec().copy()
    feature_spec.pop(transformed_name(LABEL_KEY))
    
    for col in feature_spec.keys():
        input_layers.append(tf.keras.Input(shape=(1,), name=col, dtype=feature_spec[col].dtype))
        feature_columns.append(tf.feature_column.numeric_column(col))
        
    # Menggabungkan seluruh input layer menjadi satu dense vector
    dnn_inputs = tf.keras.layers.DenseFeatures(feature_columns)(dict(zip(feature_spec.keys(), input_layers)))
    
    # Auto-tuning jumlah hidden layers dan units menggunakan Keras Tuner
    x = dnn_inputs
    for i in range(hp.Int('num_layers', min_value=1, max_value=3, default=2)):
        units = hp.Int(f'units_{i}', min_value=32, max_value=128, step=32, default=64)
        x = tf.keras.layers.Dense(units, activation='relu')(x)
        dropout_rate = hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.5, step=0.1, default=0.2)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        
    # Output layer untuk klasifikasi biner Churn (0 atau 1) menggunakan 2 unit Softmax
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=input_layers, outputs=outputs)
    
    # Auto-tuning learning rate untuk optimizer Adam
    lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
    )
    return model

def tuner_fn(fn_args: FnArgs):
    """Fungsi utama komponen Tuner TFX untuk hyperparameter tuning otomatis (Fixed Version)"""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    
    train_dataset = _input_fn(fn_args.train_files, tf_transform_output, batch_size=64)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, batch_size=64)
    
    # Menggunakan RandomSearch yang distabilkan untuk menghindari bug pengurutan NoneType TFX pada Windows
    tuner_obj = RandomSearch(
        hypermodel=lambda hp: _build_model(hp, tf_transform_output),
        objective='val_accuracy',
        max_trials=3,
        directory=fn_args.working_dir,
        project_name='churn_tuning'
    )
    
    # WAJIB dibungkus ke dalam TunerFnResult agar TFX mengenali properti internalnya
    return TunerFnResult(
        tuner=tuner_obj,
        fit_kwargs={
            'x': train_dataset,
            'validation_data': eval_dataset,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps
        }
    )

def run_fn(fn_args: FnArgs):
    """Fungsi utama komponen Trainer TFX untuk melatih model final dengan parameter terbaik"""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    
    train_dataset = _input_fn(fn_args.train_files, tf_transform_output, batch_size=64)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, batch_size=64)
    
    # Perbaikan Bintang 5: Membaca parameter terbaik secara universal dari konfigurasi objek hps
    from keras_tuner.engine.hyperparameters import HyperParameters
    hp = HyperParameters.from_config(fn_args.hyperparameters)
    model = _build_model(hp, tf_transform_output)
    
    # Melatih model final
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        epochs=10
    )
    
    # Menyimpan model lengkap dengan fungsi signature serving-nya demi kebutuhan TF Serving nanti
    signatures = {
        'serving_default': _get_serve_tf_examples_fn(model, tf_transform_output)
    }
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
