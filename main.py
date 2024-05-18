import mne
import numpy as np
from mne.decoding import CSP
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle

raw=mne.io.read_raw_gdf('BCICIV_2a_gdf/A04T.gdf',
                         eog=['EOG-left', 'EOG-central', 'EOG-right'])
raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])
events=mne.events_from_annotations(raw)
event_dict={
 'reject':1,
 'eye move':2,
 'eye open':3,
 'eye close':4,
 'new run':5,
 'new trial':6,
 'class 1':7,
 'class 2':8,
 'class 3':9,
 'class 4':10,

}
epochs = mne.Epochs(raw, events[0], event_id=[7,8],tmin= -0.1, tmax=0.7, preload=True)
evoked_0 = epochs['7'].average()
evoked_1 = epochs['8'].average()
#left,right
dicts={'class0':evoked_0,'class1':evoked_1,}
mne.viz.plot_compare_evokeds(dicts)


def read_data(path, low_freq, high_freq):
    raw = mne.io.read_raw_gdf(path, preload=True,
                              eog=['EOG-left', 'EOG-central', 'EOG-right'])
    raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])

    # Apply bandpass filter
    raw.filter(low_freq, high_freq, fir_design='firwin')

    raw.set_eeg_reference()
    events = mne.events_from_annotations(raw)

    # Filter events to only include left (event_id=7) and right (event_id=8) motor imagery tasks
    event_id = {'left': 7, 'right': 8}
    selected_events = [event for event in events[0] if event[2] in event_id.values()]

    epochs = mne.Epochs(raw, selected_events, event_id=event_id, on_missing='warn')
    labels = epochs.events[:, -1]
    features = epochs.get_data()

    return features, labels


# Read data and concatenate into features, labels, and groups
features, labels, groups = [], [], []
low_freq = 0.3
high_freq = 25
for i in range(1, 10):
    feature, label = read_data(f'BCICIV_2a_gdf/A0{i}T.gdf', low_freq, high_freq)
    features.append(feature)
    labels.append(label)
    groups.append([i] * len(label))

features = np.concatenate(features)
labels = np.concatenate(labels)
groups = np.concatenate(groups)


print("features shape :",features.shape)
print("labels shape :",labels.shape)
print("labels :" ,labels)
# Reshape features to have 3 dimensions
n_samples, n_channels, n_time_points = features.shape
X = features.reshape(n_samples, n_channels, n_time_points)
y=labels
# Apply CSP
csp = CSP(n_components=10, reg=None, log=True, norm_trace=False)
X_csp = csp.fit_transform(X, y)

# # Apply PCA
# pca = PCA(n_components=10)
# X_pca = pca.fit_transform(X_csp)

# Apply ICA
ica = FastICA(n_components=3)
X_ica = ica.fit_transform(X_csp)

# Standardize features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X_ica)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

print("_________________________________________")
print("Y_train shape:",y_train.shape)
print("Y_test shape:",y_test.shape)
print("X_train shape:",X_train.shape)
print("X_test shape:",X_test.shape)
print("X_test :",X_test)
print("Y_test :",y_test)
print("_________________________________________")

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

#Logistic Regression
lr = LogisticRegression(solver='lbfgs')
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression:")
print("Logistic Model Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred_lr))
plot_confusion_matrix(y_test,y_pred_lr)
# Save Logistic Regression model
with open('logistic_regression.pkl', 'wb') as f:
    pickle.dump(lr, f)
print("_________________________________________")

# SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_svm)
print("SVC Model Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred_svm))
plot_confusion_matrix(y_test,y_pred_svm)
# Save SVM model
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm_classifier, f)
print("_________________________________________")


# LDA classifier
lda_classifier = LinearDiscriminantAnalysis()
lda_classifier.fit(X_train, y_train)
y_pred_lda = lda_classifier.predict(X_test)
accuracy_lda = accuracy_score(y_test, y_pred_lda)
print("LDA:")
print("LDA Model Accuracy:", accuracy_lda)
print("Classification Report:")
print(classification_report(y_test, y_pred_lda))
plot_confusion_matrix(y_test, y_pred_lda)
# Save LDA model
with open('lda_model.pkl', 'wb') as f:
    pickle.dump(lda_classifier, f)
print("_________________________________________")




# # Load models
# with open('logistic_regression.pkl', 'rb') as f:
#     lr_model = pickle.load(f)
#
# with open('svm_model.pkl', 'rb') as f:
#     svm_model = pickle.load(f)
#
# with open('lda_model.pkl', 'rb') as f:
#     lda_model = pickle.load(f)
