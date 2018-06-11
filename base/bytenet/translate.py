from config import config
from data_loader import DataLoader, batches, batch
from translator import ByteNetTranslator
import argparse
import numpy as np
import tensorflow as tf


def sample_top(a, top_k= 10):
    i = np.argsort(a)[::-1][:top_k]
    p = a[i]
    p = p / np.sum(p)
    return np.random.choice(i, p= p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source_file', help= 'source file')
    parser.add_argument('target_file', help= 'target file')
    parser.add_argument('--top-k',        type= int, default=  5, help= 'sample from top k predictions')
    parser.add_argument('--batch-size',   type= int, default= 32, help= 'batch size')
    parser.add_argument('--bucket-quant', type= int, default= 50, help= 'bucket quantity')
    parser.add_argument('--model-path', default= None, help= 'pre-trained model path, to resume from')
    args = parser.parse_args()

    dl = DataLoader(args.source_file, args.target_file, args.bucket_quant)

    import pickle
    with open("baseline/vocab.p", "rb") as f:
        vocab = pickle.load(f)
    dl.source_vocab = vocab['source_vocab']
    dl.target_vocab = vocab['target_vocab']

    buckets, sents = dl.load_translation_data()
    print("number of buckets", len(buckets))

    config['source_vocab_size'] = len(dl.source_vocab)
    config['target_vocab_size'] = len(dl.target_vocab)

    model = ByteNetTranslator(config)
    model.build_translator()

    sess = tf.InteractiveSession()
    tf.train.Saver().restore(sess, args.model_path)

    with open("baseline/gold", 'w') as gold, open("baseline/pred", 'w') as pred:
        for bucket_size in sorted(buckets.keys()):
            print("translating bucket", bucket_size)
            bucket, st = buckets[bucket_size], sents[bucket_size]
            for i, j in batches(len(bucket), args.batch_size):
                source, target = batch(bucket[i:j])
                target = target[:, 0:1]
                for col in range(bucket_size):
                    probs = sess.run(
                        model.t_probs
                        , feed_dict = {model.t_source_sentence : source, model.t_target_sentence : target})
                    target = np.insert(
                        target
                        , target.shape[1]
                        , [sample_top(prob[-1], top_k= args.top_k) for prob in probs]
                        , axis = 1)
                for (_, t), p in zip(st[i:j], target):
                    print(t, file= gold)
                    print(dl.decode(p[1:]), file= pred)


if __name__ == '__main__':
    main()
