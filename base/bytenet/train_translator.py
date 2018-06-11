from data_loader import DataLoader, batches, batch
from config import config
from translator import ByteNetTranslator
import argparse
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source_file', help= 'source file')
    parser.add_argument('target_file', help= 'target file')
    parser.add_argument('--top-k',        type= int, default=  5, help= 'sample from top k predictions')
    parser.add_argument('--batch-size',   type= int, default= 32, help= 'batch size')
    parser.add_argument('--bucket-quant', type= int, default= 50, help= 'bucket quantity')
    parser.add_argument('--epochs',   type= int, default= 20, help= 'epochs')
    parser.add_argument('--learning-rate', type= float, default= 0.001, help= 'learning rate')
    parser.add_argument('--beta1',         type= float, default= 0.5,   help= 'momentum for adam update')
    parser.add_argument('--summary-every', type= int, default= 50, help= 'sample generator output every x steps')
    parser.add_argument('--resume-model', default= None, help= 'pre-trained model path, to resume from')
    parser.add_argument('--resume-from-bucket', type= int, default= 0, help= 'resume from bucket')
    args = parser.parse_args()

    dl = DataLoader(args.source_file, args.target_file, args.bucket_quant)
    buckets, _ = dl.load_translation_data()
    print("number of buckets", len(buckets))

    import pickle
    with open("baseline/vocab.p", "wb") as f:
        pickle.dump(dict(source_vocab= dl.source_vocab, target_vocab= dl.target_vocab), f)
    print("vocab saved")

    config['source_vocab_size'] = len(dl.source_vocab)
    config['target_vocab_size'] = len(dl.target_vocab)

    model = ByteNetTranslator(config)
    model.build_model()
    optim = tf.train.AdamOptimizer(args.learning_rate, beta1= args.beta1).minimize(model.loss)

    sess = tf.InteractiveSession()
    svr = tf.train.Saver()

    if args.resume_model:
        svr.restore(sess, args.resume_model)
    else:
        tf.global_variables_initializer().run()

    step = 0
    for epoch in range(args.epochs):
        for bucket_size in sorted(buckets.keys()):
            if epoch == 0 and bucket_size < args.resume_from_bucket: continue
            bucket = buckets[bucket_size]
            for i, j in batches(len(bucket), args.batch_size):
                source, target = batch(bucket[i:j])
                loss, _ = sess.run(
                    (model.loss, optim)
                    , feed_dict= {model.source_sentence : source, model.target_sentence : target})
                # TODO tf.summary
                print("loss {} epoch {} step {} bucket {}:{}".format(
                    loss, epoch, step, bucket_size, len(bucket)))
                step += 1
            save_path = svr.save(sess, "baseline/model/e{:02d}_b{:03d}.ckpt".format(epoch, bucket_size))


if __name__ == '__main__':
    main()
