main_data_path = '/raid/guoyangyang/vqa/'

# training set
train_set = 'train'
assert train_set in ['train', 'train+val']

# image paths
train_image_path = main_data_path + 'mscoco/train2014/'  # directory of training images
val_image_path = main_data_path + 'mscoco/val2014/'  # directory of validation images
test_image_path = main_data_path + 'mscoco/test2015/'  # directory of test images
preprocessed_path = main_data_path + 'grid-data/resnet-14x14.h5'  # path where preprocessed image features are saved to and loaded from

# vqa1.0 path
qa_path = main_data_path + 'vqa1.0/qa_path/'  # directory containing the question and annotation jsons
vocabulary_path = main_data_path +'vqa1.0/vocab.json'  # path where the used vocabularies for question and answers are saved to
question_type_path = main_data_path + 'vqa1.0/qtype.json' # question type with their corresponding answers

# pre-trained word embedding path
word_embedding_path = main_data_path + 'word_embed/'

# test result path (for evaluation on the challenge website)
results = './results_test/results'

# task type, we only consider open-ended
task = 'OpenEnded'
dataset = 'mscoco'

# preprocess config
preprocess_batch_size = 16
image_size = 448  # scale shorter end of image to this size and centre crop
output_size = image_size // 32  # size of the feature maps after processing through a network
output_features = 2048  # number of feature maps thereof
central_fraction = 0.875  # only take this much of the centre when scaling and centre cropping

# training config
epochs = 50
batch_size = 128
initial_lr = 1e-3  # default Adam lr
lr_halflife = 50000  # in iterations
data_workers = 8
max_answers = 3000
max_question_len = 15
