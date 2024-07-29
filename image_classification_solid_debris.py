import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as ET
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical, Sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model  # Add this import statement

from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
# Assuming an optional XML file that might contain additional data or validation info for images
def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    annotations = []
    for annotation in root.findall('.//annotation'):
        filename = annotation.find('filename').text
        objects = annotation.findall('.//object')
        # Initialize class label as None; it could also be a default label
        class_label = None
        for obj in objects:
            obj_name = obj.find('name').text
            # Determine the label based on object names
            # This example prioritizes 'debris' if present, otherwise 'solid'
            if obj_name == 'debris':
                class_label = 'debris'
                break  # Stop looking if 'debris' is found
            elif obj_name == 'solid':
                class_label = 'solid'
        # Append the filename and determined class label to annotations
        # Skip if no recognizable object was found (class_label remains None)
        if class_label is not None:
            annotations.append((filename, class_label))
    return annotations


# Load and preprocess image data based on class directories
def load_and_preprocess_data(data_dir, classes, target_shape=(224, 224), xml_file=None):
    additional_info = parse_xml(xml_file) if xml_file else {}
    data = []
    labels = []
    
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith('.jpg'):
                # Optional: Use additional info from XML for filtering or annotation
                if filename in additional_info:
                    # Example condition, customize as needed
                    if additional_info[filename] == "Unspecified":
                        continue
                
                file_path = os.path.join(class_dir, filename)
                img = load_img(file_path, target_size=target_shape)
                img_array = img_to_array(img) / 255.0
                data.append(img_array)
                labels.append(i)
    
    return np.array(data), np.array(labels)

data_dir = "D:/Development/model_enkaz/new_model/dataset"
classes = ["solid", "debris"]
xml_file = "D:/Development/model_enkaz/new_model/dataset/solid_debris.xml"  # Optional XML file for additional data or validation

# Load data
data, labels = load_and_preprocess_data(data_dir, classes, xml_file=xml_file)
labels = to_categorical(labels, num_classes=len(classes))

# Split data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# The model building, compiling, training, and evaluation steps remain unchanged.


# Model creation using MobileNetV2 as base
def build_model(input_shape, num_classes):
    base_model = MobileNetV2(include_top=False, input_shape=input_shape, weights='imagenet', pooling='avg')
    base_model.trainable = False  # Freeze base model layers
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

# Compile and train the model
model = build_model(X_train[0].shape, len(classes))
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test accuracy: {test_accuracy[1]}')

# Save the model
model_filename = 'image_classification_model_debris_solid'
if os.path.exists(model_filename):
    os.remove(model_filename)  # Delete the file if it exists


tf.saved_model.save(model, 'image_classification_model_debris_solid')
