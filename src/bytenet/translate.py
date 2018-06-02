import tensorflow as tf
import numpy as np
import argparse
import model_config
import data_loader
from ByteNet import translator
import utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket_quant', type=int, default=50,
                       help='Learning Rate')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Pre-Trained Model Path, to resume from')
    parser.add_argument('--source_file', type=str, default='Data/MachineTranslation/news-commentary-v11.de-en.de',
                       help='Source File')
    parser.add_argument('--target_file', type=str, default='Data/MachineTranslation/news-commentary-v11.de-en.en',
                       help='Target File')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Sample from top k predictions')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch Size')
    args = parser.parse_args()

    data_loader_options = {
        'model_type' : 'translation',
        'source_file' : args.source_file,
        'target_file' : args.target_file,
        'bucket_quant' : args.bucket_quant,
    }

    dl = data_loader.Data_Loader(data_loader_options)

    import pickle
    with open("Data/Models/translation_model/vocab.p", "rb") as f:
        vocab = pickle.load(f)
    dl.source_vocab = vocab['source_vocab']
    dl.target_vocab = vocab['target_vocab']

    buckets, source_vocab, target_vocab = dl.load_translation_data()
    print("Number Of Buckets", len(buckets))
    indices = dl.indices # hack to keep track of original sentence

    config = model_config.translator_config
    model_options = {
        'source_vocab_size' : len(source_vocab),
        'target_vocab_size' : len(target_vocab),
        'residual_channels' : config['residual_channels'],
        'decoder_dilations' : config['decoder_dilations'],
        'encoder_dilations' : config['encoder_dilations'],
        'decoder_filter_width' : config['decoder_filter_width'],
        'encoder_filter_width' : config['encoder_filter_width'],
    }

    translator_model = translator.ByteNet_Translator( model_options )
    translator_model.build_translator()

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, args.model_path)

    def batches(size, batch_size):
        i = 0
        while i < size:
            yield i, i + batch_size
            i += batch_size

    with open("Data/MachineTranslation/gold", 'w') as gold, open("Data/MachineTranslation/pred", 'w') as pred:
        for bucket_size in sorted(buckets.keys()):
            print("translating bucket", bucket_size)
            bucket = buckets[bucket_size]
            indice = indices[bucket_size]
            for i, j in batches(len(bucket), args.batch_size):
                source, target = dl.get_batch_from_pairs(bucket[i:j])
                target = target[:, 0:1]
                for col in range(bucket_size):
                    probs = sess.run(translator_model.t_probs,
                                     feed_dict = {
                                         translator_model.t_source_sentence : source,
                                         translator_model.t_target_sentence : target,
                                     })
                    curr_preds = []
                    for bi in range(probs.shape[0]):
                        pred_word = utils.sample_top(probs[bi][-1], top_k = args.top_k )
                        curr_preds.append(pred_word)
                    target = np.insert(target, target.shape[1], curr_preds, axis = 1)
                for k, p in zip(indice[i:j], target):
                    print(dl.target_lines[k], file= gold)
                    print(dl.inidices_to_string(p[1:], target_vocab), file= pred)


if __name__ == '__main__':
    main()
