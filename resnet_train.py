from resnet import * 
import tensorflow as tf

MOMENTUM = 0.9

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/tmp/resnetearly_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_float('learning_rate', 0.01, "learning rate.")
tf.app.flags.DEFINE_integer('batch_size', 64, "batch size")
tf.app.flags.DEFINE_integer('max_steps', 500000, "max steps")
tf.app.flags.DEFINE_boolean('resume', False,
                            'resume from latest saved state')
tf.app.flags.DEFINE_boolean('minimal_summaries', False,
                            'produce fewer summaries to save HD space')


def top_k_error(predictions, labels, k):
    batch_size = float(FLAGS.batch_size) #tf.shape(predictions)[0]
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
    num_correct = tf.reduce_sum(in_top1)
    return (batch_size - num_correct) / batch_size

def train(is_training, logits, logits2list, images, labels, varlist, varlist2):
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    val_step = tf.get_variable('val_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    loss_, loss2 = loss(logits, logits2list, labels)

    logitsend = logits
    numearlyresults = []
    with tf.device('/cpu:0'):
        confthreshvar = tf.Variable(1.0, name='confth')

    for l in reversed(logits2list):
        softmax = tf.nn.softmax(l)
        conf1 = tf.reduce_max(softmax,1)
        confthresh = tf.fill(conf1.get_shape(), confthreshvar)
        boolmask = tf.greater_equal(conf1,confthresh)
        numearlyresult = tf.reduce_sum(tf.cast(boolmask, tf.float32))
        numearlyresults.append(numearlyresult)
        boolmask = tf.expand_dims(boolmask,1)
        boolmask = tf.concat(1,[boolmask for i in range(0,10)])


        logits = tf.select(boolmask,l,logits)


    predictions = tf.nn.softmax(logits)
    predictionsend = tf.nn.softmax(logitsend)

    top1_error = top_k_error(predictions, labels, 1)
    top1_error_end = top_k_error(predictionsend, labels, 1)


    print(len(numearlyresults))
    # loss_avg
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, ema.apply([loss_]))
    tf.scalar_summary('loss_avg', ema.average(loss_))

    ema2 = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, ema2.apply([loss2]))
    tf.scalar_summary('loss2_avg', ema2.average(loss2))

    # validation stats
    ema = tf.train.ExponentialMovingAverage(0.9, val_step)
    val_op = tf.group(val_step.assign_add(1), ema.apply([top1_error]))
    top1_error_avg = ema.average(top1_error)

    ema = tf.train.ExponentialMovingAverage(0.9, val_step)
    val_op_end = tf.group(val_step.assign_add(1), ema.apply([top1_error_end]))
    top1_error_avg_end = ema.average(top1_error_end)

    decay_steps = 30000
    lr = tf.train.exponential_decay(FLAGS.learning_rate,
                                  global_step,
                                  decay_steps,
                                  0.5,
                                  staircase=True)

    tf.scalar_summary('val_top1_error_avg', top1_error_avg)
    tf.scalar_summary('val_top1_error_avg_end', top1_error_avg_end)

    tf.scalar_summary('learning_rate', lr)

    opt = tf.train.MomentumOptimizer(lr,MOMENTUM)
    grads1 = opt.compute_gradients(loss_, varlist)
    grads2 = opt.compute_gradients(loss2, varlist2)
    grads1 = [(tf.clip_by_value(gv[0], -0.3, 0.3), gv[1]) for gv in grads1]
    grads2 = [(tf.clip_by_value(gv[0], -1., 1.), gv[1]) for gv in grads2]
    grads = grads1 + grads2
    for grad, var in grads:
        if grad is not None and not FLAGS.minimal_summaries:
            tf.histogram_summary(var.op.name + '/gradients', grad)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    if not FLAGS.minimal_summaries:
        # Display the training images in the visualizer.
        tf.image_summary('images', images)

        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)

    batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

    saver = tf.train.Saver(tf.all_variables())

    summary_op = tf.merge_all_summaries()

    init = tf.initialize_all_variables()

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(init)

    value = 0.9
    sess.run(confthreshvar.assign(value))

    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    if FLAGS.resume:
        latest = tf.train.latest_checkpoint(FLAGS.train_dir)
        if not latest:
            print "No checkpoint to continue from in", FLAGS.train_dir
            sys.exit(1)
        print "resume", latest
        saver.restore(sess, latest)
    for x in xrange(FLAGS.max_steps + 1):
        start_time = time.time()

        step = sess.run(global_step)
        i = [train_op, loss_, loss2]

        write_summary = step % 100 and step > 1
        if write_summary:
            i.append(summary_op)

        o = sess.run(i, { is_training: True })

        loss_value = o[1]
        loss_value2 = o[2]


        duration = time.time() - start_time

        #assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 5 == 0:
            examples_per_sec = FLAGS.batch_size / float(duration)
            format_str = ('step %d, loss = %.3f, loss2 = %.3f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (step, loss_value, loss_value2, examples_per_sec, duration))

        if write_summary:
            summary_str = o[3]
            summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step > 1 and step % 100 == 0:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=global_step)

        # Run validation periodically
        if step > 1 and step % 300 == 0:
            res = sess.run([val_op,val_op_end, top1_error, top1_error_end] + numearlyresults, { is_training: False })
            top1_error_value = res[2]
            top1_error_value_end = res[3]
            earlyresults = res[4:]
            sumearly = 0
            for i in range(0, len(numearlyresults)-1):
                earlyresults[i] = earlyresults[i] - earlyresults[i+1]
                sumearly += earlyresults[i]
            sumearly += earlyresults[-1]
            sumend = FLAGS.batch_size - sumearly
            print('top1 error %.2f, top1 error end: %.2f' % (top1_error_value, top1_error_value_end))
            print('ToEnd: %d, %d, %d, %d, %d, %d, %d, %d, %d, %d' % (sumend, earlyresults[0],earlyresults[1], earlyresults[2], earlyresults[3],earlyresults[4],
                  earlyresults[5],earlyresults[6],earlyresults[7],earlyresults[8]))


