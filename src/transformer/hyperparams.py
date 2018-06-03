class Hyperparams:
    '''Hyperparameters'''
     # data
    source_train = 'corpora/train.tags.de-en.de'
    target_train = 'corpora/train.tags.de-en.en'
    source_test = 'corpora/IWSLT16.TED.tst2014.de-en.de.xml'
    target_test = 'corpora/IWSLT16.TED.tst2014.de-en.en.xml'
    # training
    batch_size = 32 # alias = N
    lr = 0.0001 # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'baseline' # log directory
    # model
    max_len = 10 # Maximum number of words in a sentence. alias = T.
    min_cnt = 20 # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512 # alias = C
    num_blocks = 6 # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False # If True, use sinusoid. If false, positional embedding.

    # mock the first 12 lines
    source_train = '../../mock/train.nen'
    target_train = '../../mock/train.sen'
    source_test  = '../../mock/test.nen'
    target_test  = '../../mock/test.sen'
    max_len = 24 # max len in mock
    min_cnt = 2
    batch_size = 16
    num_epochs = 2

    # source_train = '../../data/train.nen'
    # target_train = '../../data/train.sen'
    # source_test  = '../../data/test.nen'
    # target_test  = '../../data/test.sen'
    # max_len = 65 # max len in data/test
    # min_cnt = 5
    # batch_size = 32
    # num_epochs = 20
