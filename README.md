
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from mlxtend.classifier import StackingCVClassifier

import xgboost as xgb
import lightgbm as lgb

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

# Load the dataset
df = pd.read_csv('/content/kidney_urine_with_id.csv')

print("✅ 'id' column added and new file saved as 'kidney_urine_with_id.csv'.")

# Load the dataset
df = pd.read_csv('kidney_urine_with_id.csv')
# drop the 'id' column
df.drop('id', axis=1, inplace=True)

df.columns

data_without_target = df.drop(columns=['target'])

# Distribution plots
for column in data_without_target.columns:
    plt.figure()
    sns.histplot(data=data_without_target, x=column, kde=True, color='blue')
    plt.title(f'Distribution of {column}')
    plt.show()

# Scatterplot matrix
sns.set(style='whitegrid')
sns.pairplot(df, hue='target', diag_kind='kde', markers=['o', 's'], plot_kws={'alpha': 0.7})
plt.show()

# Box plots and swarm plots
plt.figure(figsize=(15, 15))
for i, column in enumerate(data_without_target.columns):
    plt.subplot(len(data_without_target.columns), 2, i + 1)
    sns.boxplot(x='target', y=column, data=df, palette='coolwarm')
    sns.swarmplot(x='target', y=column, data=df, color='black', alpha=0.4)
    plt.title(f'{column} by Target')
plt.tight_layout()
plt.show()

target_counts = df['target'].value_counts(normalize=True)

ax = target_counts.plot(kind='bar', stacked=True, figsize=(10, 6))

ax.set_xlabel('Target', fontsize=16)
ax.set_ylabel('Proportion', fontsize=16)
ax.set_title('Proportion of Kidney Stone Presence in the Dataset', fontsize=20, fontweight='bold')
plt.xticks(ticks=[0, 1], labels=['False', 'True'], rotation=0, fontsize=14)
plt.yticks(fontsize=12)

for i, v in enumerate(target_counts):
    ax.text(i - 0.1, v + 0.01, f'{v:.2f}', fontsize=14, color='black')

plt.show()

# compute the correlation matrix with the target column
corr = df.corr()

# extract the correlations with the target column
target_corr = corr['target'].drop('target')

# create a heatmap of the correlations with the target column
sns.set(font_scale=1.2)
sns.set_style("white")
sns.set_palette("PuBuGn_d")
sns.heatmap(target_corr.to_frame(), cmap="coolwarm", annot=True, fmt='.2f')
plt.title('Correlation with Target Column')
plt.show()

# Calculate estimated urine volume
df['urine_volume'] = (1000 * df['gravity'] * df['osmo']) / (18 * 1.001)

# Specific gravity to calcium ratio
df['specific_gravity_calcium_ratio'] = df['gravity'] / df['calc']

# Calcium to conductivity product
df['calcium_conductivity_ratio'] = df['calc'] / df['cond']

# Calcium and pH product
df['calcium_pH_interaction'] = df['calc'] * df['ph']

# Urea and pH product
df['urea_pH_interaction'] = df['urea'] * df['ph']

# Osmolarity and calcium product
df['osmolarity_calcium_interaction'] = df['osmo'] * df['calc']

### **NEW: Sugar-Based Feature Calculations**
# Sugar-Specific Gravity Ratio
df['sugar_gravity_ratio'] = df['gravity'] / (df['sugar'] + 1)  # Avoid division by zero

# Sugar-Osmolality Interaction
df['sugar_osmo_interaction'] = df['osmo'] * df['sugar']

# Sugar-Conductivity Ratio
df['sugar_cond_ratio'] = df['cond'] / (df['sugar'] + 1)  # Avoid division by zero

# Sugar-pH Product
df['sugar_pH_product'] = df['sugar'] * df['ph']

# Sugar-Urea Interaction
df['sugar_urea_interaction'] = df['sugar'] * df['urea']

### **NEW: Sugar Interpretation**
df['sugar_interpretation'] = df['sugar'].map({
    0: 'Low Sugar - Normal',
    1: 'Medium Sugar - Possible Hyperglycemia',
    2: 'High Sugar - Potential Diabetes Risk'
})

# Display first few rows
df.head()

# Drop non-numeric columns before correlation
numeric_df = df.select_dtypes(include=['number'])  # Select only numeric columns

# Compute correlation matrix
corr = numeric_df.corr()

# Sort correlation with target
target_corr = corr['target'].drop('target').sort_values(ascending=False)

# Plot heatmap
plt.figure(figsize=(12,8))
sns.set(font_scale=1)
sns.set_style("white")
sns.set_palette("PuBuGn_d")
sns.heatmap(target_corr.to_frame(), cmap="coolwarm", annot=True, fmt='.2f')
plt.title('Correlation with Target Column')
plt.show()

# Ensure only numeric columns are used
X = df.select_dtypes(include=['number']).drop(columns=['target'])
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Show accuracy
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:", classification_report(y_test, y_pred))
print("Confusion Matrix:", confusion_matrix(y_test, y_pred))

# Check feature importance
feature_importances = clf.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
print(feature_importance_df)

import pandas as pd

# Ensure model is trained
rf = RandomForestClassifier()  # Replace with your model
rf.fit(X_train, y_train)  # Train the model

# Extract feature importance
feature_importance = rf.feature_importances_  # Works for tree-based models

# Create DataFrame with feature names and importance values
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importance
})

# Sort and select top features
num_features = 5  # Choose based on experimentation
selected_features = feature_importance_df.nlargest(num_features, 'Importance')['Feature'].values

# Create a new DataFrame with only selected features
X_top = X[selected_features]

print("Selected Features:", selected_features)

# Data Preprocessing

# Handle class imbalance
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X[selected_features], y)

# Use SMOTE-resampled data for training
X_train, X_val, y_train, y_val = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Prediction Model (Regression)
models = [
    ('SVC', SVC(random_state=42)),
    ('RandomForestClassifier', RandomForestClassifier(random_state=42)),
    ('XGBoost', xgb.XGBClassifier(random_state=42)),
    ('LightGBM', lgb.LGBMClassifier(random_state=42)),
]

# Train and evaluate the models
for name, model in models:
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_val_scaled)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"{name} accuracy: {accuracy:.4f}")

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# ✅ Define base classifiers
svc = SVC(probability=True)  # Ensure probability=True for stacking
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gbc = GradientBoostingClassifier(n_estimators=100, random_state=42)
mlp = MLPClassifier(max_iter=500, random_state=42)  # Ensure it trains well

# ✅ Define meta-model
meta_model = RandomForestClassifier(n_estimators=100, random_state=42)

# ✅ Define stacking classifier
stacking_classifier = StackingClassifier(
    estimators=[('svc', svc), ('rf', rf), ('gbc', gbc), ('mlp', mlp)],
    final_estimator=meta_model
)

# Fit the model
stacking_classifier.fit(X_train_scaled, y_train)

# Predictions
y_val_pred = stacking_classifier.predict(X_val_scaled)

# Accuracy
stacking_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Stacking Classifier accuracy: {stacking_accuracy:.4f}")

def create_nn_model(input_dim):
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),  # More neurons
        layers.BatchNormalization(),  # Normalize activations
        layers.Dropout(0.3),  # Prevent overfitting

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

optimizer = keras.optimizers.Adam(learning_rate=0.0005)  # Slower, more stable learning

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

# Create the neural network model
nn_model = create_nn_model(X_train_scaled.shape[1])

nn_history = nn_model.fit(
    X_train_scaled, y_train, validation_data=(X_val_scaled, y_val),
    epochs=300, batch_size=32, verbose=1, callbacks=[early_stop]
)

print(y_train.value_counts())

from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# Get final training & validation accuracy
train_accuracy = nn_history.history['accuracy'][-1]  # Last epoch's training accuracy
val_accuracy = nn_history.history.get('val_accuracy', [None])[-1]  # Last epoch's validation accuracy

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ✅ Create a SIMPLER neural network
def create_optimized_nn(input_dim):
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(1, activation='sigmoid')
    ])

    optimizer = Adam(learning_rate=0.0005)  # ✅ Lower learning rate
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ✅ Train the optimized model
optimized_nn = create_optimized_nn(X_train_scaled.shape[1])

early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-5)

optimized_nn_history = optimized_nn.fit(
    X_train_scaled, y_train, validation_data=(X_val_scaled, y_val),
    epochs=300, batch_size=32, verbose=1,  # ✅ Larger batch size
    callbacks=[early_stopping, reduce_lr]
)

# ✅ Show accuracy
optimized_nn_accuracy = optimized_nn_history.history['val_accuracy'][-1]
print(f"Optimized Neural Network Accuracy: {optimized_nn_accuracy:.4f}")

# Load the test dataset
test_df = pd.read_csv('kidney_urine_with_id.csv')

# Calculate estimated urine volume
test_df['urine_volume'] = (1000 * test_df['gravity'] * test_df['osmo']) / (18 * 1.001)

# Specific gravity to calcium ratio
test_df['specific_gravity_calcium_ratio'] = test_df['gravity'] / test_df['calc']

# Calcium to conductivity product
test_df['calcium_conductivity_ratio'] = test_df['calc'] / test_df['cond']

# Calcium and pH product
test_df['calcium_pH_interaction'] = test_df['calc'] * test_df['ph']

# Urea and pH product
test_df['urea_pH_interaction'] = test_df['urea'] * test_df['ph']

# Osmolarity and calcium product
test_df['osmolarity_calcium_interaction'] = test_df['osmo'] * test_df['calc']

### **NEW: Sugar-Based Feature Calculations**
# Sugar-Specific Gravity Ratio
test_df['sugar_gravity_ratio'] = test_df['gravity'] / (test_df['sugar'] + 1)  # Avoid division by zero

# Sugar-Osmolality Interaction
test_df['sugar_osmo_interaction'] = test_df['osmo'] * test_df['sugar']

# Sugar-Conductivity Ratio
test_df['sugar_cond_ratio'] = test_df['cond'] / (test_df['sugar'] + 1)  # Avoid division by zero

# Sugar-pH Product
test_df['sugar_pH_product'] = test_df['sugar'] * test_df['ph']

# Sugar-Urea Interaction
test_df['sugar_urea_interaction'] = test_df['sugar'] * test_df['urea']

### **NEW: Sugar Interpretation**
test_df['sugar_interpretation'] = test_df['sugar'].map({
    0: 'Low Sugar - Normal',
    1: 'Medium Sugar - Possible Hyperglycemia',
    2: 'High Sugar - Potential Diabetes Risk'
})

# Display first few rows to verify
test_df.head()

# Preprocess the test dataset
X_test = test_df[selected_features]

# Scale the test dataset
X_test_scaled = scaler.transform(X_test)

# Make predictions using the optimized neural network model
y_test_pred = optimized_nn.predict(X_test_scaled)

# Create a new DataFrame with ID column and predicted probabilities
result_df = pd.DataFrame({'id': test_df['id'], 'predicted_probability': y_test_pred.reshape(-1)})

result_df.tail(20)

# Save the result DataFrame as a CSV file
result_df.to_csv('submission.csv', index=False)
