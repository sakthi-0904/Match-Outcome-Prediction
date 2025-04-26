import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Your dataset
# X = features
# y = target

# Example dummy data:
X = pd.DataFrame({
    'B365H': [2.15, 1.95, 2.75, 2.50, 2.00],
    'B365D': [3.40, 3.25, 3.10, 3.00, 3.20],
    'B365A': [3.05, 4.20, 3.45, 3.60, 3.50],
    'goal_diff': [1, 0, 2, -1, 0],
    'home_advantage': [1, 0, 1, 0, 0]
})
y = ['Home Win', 'Draw', 'Home Win', 'Away Win', 'Draw']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train model
model = RandomForestClassifier()
model.fit(X, y_encoded)

# Save model
pickle.dump(model, open('model.pkl', 'wb'))

# Save label encoder
pickle.dump(le, open('label_encoder.pkl', 'wb'))
