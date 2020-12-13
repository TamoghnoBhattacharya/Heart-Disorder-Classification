import data_to_csv
import data_preprocessing
import train

PATH = 'Path of MAT File here'
data_to_csv.new_mat_to_csv(PATH)
data_preprocessing.data_clean(PATH)
data = pd.read_csv('data/data_preprocessed.csv')
model = load_model('models/cnn_model_1.h5')
pred = model.predict_classes(data)
print(names[pred[0]])
