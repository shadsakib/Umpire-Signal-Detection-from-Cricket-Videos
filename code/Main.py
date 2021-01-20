from Train import Train

params = {
    'EPOCHS': 40,
    'BATCH_SIZE': 4,
    'LEARNING_RATE': 0.001, # range 1>lr > 0.000001
    'DIVIDE_LEARNING_RATE_AT': [],      # At what epochs, learning rate should be divided. Starts at 0.

    'DATA_PATH': '/content/VideoClassification/VideosResized224',
    'TRAIN_LABELS': '/content/VideoClassification/train.csv',
    'VAL_LABELS': '/content/VideoClassification/valid.csv',
    'TEST_LABELS': '/content/VideoClassification/test.csv',
    'LABEL_TEXT': '/content/VideoClassification/labels.csv',

    'SEQ_LEN': 60,                      # Number of frames in each video clip
    'IMG_WIDTH': 224,                   # Width of video
    'IMG_HEIGHT': 224,                  # Height of video

    # List all the labels you wish to train on.
    'REQD_LABELS': ['Four', 'NoBall', 'Out',
                    'Six', 'Wide', 'NoSignal']
}

t = Train(params)
t.train()
t.test()
