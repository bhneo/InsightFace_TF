import tensorflow as tf
import argparse
from config import config as cfg
from data.mx2tfrecords import parse_function
import os
from nets.L_Resnet_E_IR_fix_issue9 import get_resnet
from losses.face_losses import arcface_loss
from tensorflow.core.protobuf import config_pb2
import time
from data.eval_data_reader import load_bin
from verification import ver_test


def parse_args():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--net_depth', default=100, help='resnet depth, default is 50')
    parser.add_argument('--epoch', default=100000, help='epoch to train the network')
    parser.add_argument('--batch_size', default=32, help='batch size to train network')
    parser.add_argument('--lr_steps', default=[40000, 60000, 80000], help='learning rate to train network')
    parser.add_argument('--momentum', default=0.9, help='learning alg momentum')
    parser.add_argument('--weight_deacy', default=5e-4, help='learning alg momentum')
    # parser.add_argument('--eval_datasets', default=['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30'], help='evluation datasets')
    parser.add_argument('--eval_datasets', default=['lfw'], help='evluation datasets')
    parser.add_argument('--eval_db_path', default='./datasets/faces_ms1m_112x112', help='evluate datasets base path')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--num_output', default=85164, help='the image size')
    parser.add_argument('--tfrecords_file_path', default='./datasets/tfrecords', type=str,
                        help='path to the output of tfrecords file path')
    parser.add_argument('--summary_path', default='./output/summary', help='the summary file save path')
    parser.add_argument('--ckpt_path', default='./output/ckpt', help='the ckpt file save path')
    parser.add_argument('--log_file_path', default='./output/logs', help='the ckpt file save path')
    parser.add_argument('--saver_maxkeep', default=100, help='tf.train.Saver max keep ckpt files')
    parser.add_argument('--buffer_size', default=10000, help='tf dataset api buffer size')
    parser.add_argument('--log_device_mapping', default=False, help='show device placement log')
    parser.add_argument('--summary_interval', default=300, help='interval to save summary')
    parser.add_argument('--ckpt_interval', default=10000, help='intervals to save ckpt file')
    parser.add_argument('--validate_interval', default=2000, help='intervals to save ckpt file')
    parser.add_argument('--show_info_interval', default=20, help='intervals to save ckpt file')
    args = parser.parse_args()
    return args


def get_symbol(args):
    embedding = eval(config.net_name).get_symbol()
    gt_label = mx.symbol.Variable('softmax_label')
    if cfg.loss_name == 'softmax':
        _weight = mx.symbol.Variable("fc7_weight", shape=(config.num_classes, config.emb_size), lr_mult=config.fc7_lr_mult, wd_mult=config.fc7_wd_mult, init=mx.init.Normal(0.01))
        if config.fc7_no_bias:
            fc7 = mx.sym.FullyConnected(data=embedding, weight = _weight, no_bias = True, num_hidden=config.num_classes, name='fc7')
        else:
            _bias = mx.symbol.Variable('fc7_bias', lr_mult=2.0, wd_mult=0.0)
            fc7 = mx.sym.FullyConnected(data=embedding, weight = _weight, bias = _bias, num_hidden=config.num_classes, name='fc7')
    elif config.loss_name=='margin_softmax':
        _weight = mx.symbol.Variable("fc7_weight", shape=(config.num_classes, config.emb_size),
            lr_mult=config.fc7_lr_mult, wd_mult=config.fc7_wd_mult, init=mx.init.Normal(0.01))
        s = config.loss_s
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')*s
        fc7 = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=config.num_classes, name='fc7')
        if config.loss_m1!=1.0 or config.loss_m2!=0.0 or config.loss_m3!=0.0:
            if config.loss_m1==1.0 and config.loss_m2==0.0:
                s_m = s*config.loss_m3
                gt_one_hot = mx.sym.one_hot(gt_label, depth = config.num_classes, on_value = s_m, off_value = 0.0)
                fc7 = fc7-gt_one_hot
            else:
                zy = mx.sym.pick(fc7, gt_label, axis=1)
                cos_t = zy/s
                t = mx.sym.arccos(cos_t)
                if config.loss_m1!=1.0:
                    t = t*config.loss_m1
                if config.loss_m2>0.0:
                    t = t+config.loss_m2
                body = mx.sym.cos(t)
                if config.loss_m3>0.0:
                    body = body - config.loss_m3
                new_zy = body*s
                diff = new_zy - zy
                diff = mx.sym.expand_dims(diff, 1)
                gt_one_hot = mx.sym.one_hot(gt_label, depth = config.num_classes, on_value = 1.0, off_value = 0.0)
                body = mx.sym.broadcast_mul(gt_one_hot, diff)
                fc7 = fc7+body
    out_list = [mx.symbol.BlockGrad(embedding)]
    softmax = mx.symbol.SoftmaxOutput(data=fc7, label = gt_label, name='softmax', normalization='valid')
    out_list.append(softmax)
    if config.ce_loss:
        #ce_loss = mx.symbol.softmax_cross_entropy(data=fc7, label = gt_label, name='ce_loss')/args.per_batch_size
        body = mx.symbol.SoftmaxActivation(data=fc7)
        body = mx.symbol.log(body)
        _label = mx.sym.one_hot(gt_label, depth = config.num_classes, on_value = -1.0, off_value = 0.0)
        body = body*_label
        ce_loss = mx.symbol.sum(body)/args.per_batch_size
        out_list.append(mx.symbol.BlockGrad(ce_loss))
    out = mx.symbol.Group(out_list)
    return out


def train_net(args):
    ctx = []
    cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
    if len(cvd)>0:
      for i in xrange(len(cvd.split(','))):
        ctx.append(mx.gpu(i))
    if len(ctx)==0:
      ctx = [mx.cpu()]
      print('use cpu')
    else:
      print('gpu num:', len(ctx))
    prefix = os.path.join(args.models_root, '%s-%s-%s'%(args.network, args.loss, args.dataset), 'model')
    prefix_dir = os.path.dirname(prefix)
    print('prefix', prefix)
    if not os.path.exists(prefix_dir):
      os.makedirs(prefix_dir)
    args.ctx_num = len(ctx)
    args.batch_size = args.per_batch_size*args.ctx_num
    args.rescale_threshold = 0
    args.image_channel = config.image_shape[2]
    config.batch_size = args.batch_size
    config.per_batch_size = args.per_batch_size

    data_dir = config.dataset_path
    path_imgrec = None
    path_imglist = None
    image_size = config.image_shape[0:2]
    assert len(image_size)==2
    assert image_size[0]==image_size[1]
    print('image_size', image_size)
    print('num_classes', config.num_classes)
    path_imgrec = os.path.join(data_dir, "train.rec")

    print('Called with argument:', args, config)
    data_shape = (args.image_channel,image_size[0],image_size[1])
    mean = None

    begin_epoch = 0
    if len(args.pretrained)==0:
      arg_params = None
      aux_params = None
      sym = get_symbol(args)
      if config.net_name=='spherenet':
        data_shape_dict = {'data' : (args.per_batch_size,)+data_shape}
        spherenet.init_weights(sym, data_shape_dict, args.num_layers)
    else:
      print('loading', args.pretrained, args.pretrained_epoch)
      _, arg_params, aux_params = mx.model.load_checkpoint(args.pretrained, args.pretrained_epoch)
      sym = get_symbol(args)

    if config.count_flops:
      all_layers = sym.get_internals()
      _sym = all_layers['fc1_output']
      FLOPs = flops_counter.count_flops(_sym, data=(1,3,image_size[0],image_size[1]))
      _str = flops_counter.flops_str(FLOPs)
      print('Network FLOPs: %s'%_str)

    #label_name = 'softmax_label'
    #label_shape = (args.batch_size,)
    model = mx.mod.Module(
        context       = ctx,
        symbol        = sym,
    )
    val_dataiter = None

    if config.loss_name.find('triplet')>=0:
      from triplet_image_iter import FaceImageIter
      triplet_params = [config.triplet_bag_size, config.triplet_alpha, config.triplet_max_ap]
      train_dataiter = FaceImageIter(
          batch_size           = args.batch_size,
          data_shape           = data_shape,
          path_imgrec          = path_imgrec,
          shuffle              = True,
          rand_mirror          = config.data_rand_mirror,
          mean                 = mean,
          cutoff               = config.data_cutoff,
          ctx_num              = args.ctx_num,
          images_per_identity  = config.images_per_identity,
          triplet_params       = triplet_params,
          mx_model             = model,
      )
      _metric = LossValueMetric()
      eval_metrics = [mx.metric.create(_metric)]
    else:
      from image_iter import FaceImageIter
      train_dataiter = FaceImageIter(
          batch_size           = args.batch_size,
          data_shape           = data_shape,
          path_imgrec          = path_imgrec,
          shuffle              = True,
          rand_mirror          = config.data_rand_mirror,
          mean                 = mean,
          cutoff               = config.data_cutoff,
          color_jittering      = config.data_color,
          images_filter        = config.data_images_filter,
      )
      metric1 = AccMetric()
      eval_metrics = [mx.metric.create(metric1)]
      if config.ce_loss:
        metric2 = LossValueMetric()
        eval_metrics.append( mx.metric.create(metric2) )

    if config.net_name=='fresnet' or config.net_name=='fmobilefacenet':
      initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2) #resnet style
    else:
      initializer = mx.init.Xavier(rnd_type='uniform', factor_type="in", magnitude=2)
    #initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2) #resnet style
    _rescale = 1.0/args.ctx_num
    opt = optimizer.SGD(learning_rate=args.lr, momentum=args.mom, wd=args.wd, rescale_grad=_rescale)
    _cb = mx.callback.Speedometer(args.batch_size, args.frequent)

    ver_list = []
    ver_name_list = []
    for name in config.val_targets:
      path = os.path.join(data_dir,name+".bin")
      if os.path.exists(path):
        data_set = verification.load_bin(path, image_size)
        ver_list.append(data_set)
        ver_name_list.append(name)
        print('ver', name)



    def ver_test(nbatch):
      results = []
      for i in range(len(ver_list)):
        acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(ver_list[i], model, args.batch_size, 10, None, None)
        print('[%s][%d]XNorm: %f' % (ver_name_list[i], nbatch, xnorm))
        #print('[%s][%d]Accuracy: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc1, std1))
        print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc2, std2))
        results.append(acc2)
      return results



    highest_acc = [0.0, 0.0]  #lfw and target
    #for i in xrange(len(ver_list)):
    #  highest_acc.append(0.0)
    global_step = [0]
    save_step = [0]
    lr_steps = [int(x) for x in args.lr_steps.split(',')]
    print('lr_steps', lr_steps)
    def _batch_callback(param):
      #global global_step
      global_step[0]+=1
      mbatch = global_step[0]
      for step in lr_steps:
        if mbatch==step:
          opt.lr *= 0.1
          print('lr change to', opt.lr)
          break

      _cb(param)
      if mbatch%1000==0:
        print('lr-batch-epoch:',opt.lr,param.nbatch,param.epoch)

      if mbatch>=0 and mbatch%args.verbose==0:
        acc_list = ver_test(mbatch)
        save_step[0]+=1
        msave = save_step[0]
        do_save = False
        is_highest = False
        if len(acc_list)>0:
          #lfw_score = acc_list[0]
          #if lfw_score>highest_acc[0]:
          #  highest_acc[0] = lfw_score
          #  if lfw_score>=0.998:
          #    do_save = True
          score = sum(acc_list)
          if acc_list[-1]>=highest_acc[-1]:
            if acc_list[-1]>highest_acc[-1]:
              is_highest = True
            else:
              if score>=highest_acc[0]:
                is_highest = True
                highest_acc[0] = score
            highest_acc[-1] = acc_list[-1]
            #if lfw_score>=0.99:
            #  do_save = True
        if is_highest:
          do_save = True
        if args.ckpt==0:
          do_save = False
        elif args.ckpt==2:
          do_save = True
        elif args.ckpt==3:
          msave = 1

        if do_save:
          print('saving', msave)
          arg, aux = model.get_params()
          if config.ckpt_embedding:
            all_layers = model.symbol.get_internals()
            _sym = all_layers['fc1_output']
            _arg = {}
            for k in arg:
              if not k.startswith('fc7'):
                _arg[k] = arg[k]
            mx.model.save_checkpoint(prefix, msave, _sym, _arg, aux)
          else:
            mx.model.save_checkpoint(prefix, msave, model.symbol, arg, aux)
        print('[%d]Accuracy-Highest: %1.5f'%(mbatch, highest_acc[-1]))
      if config.max_steps>0 and mbatch>config.max_steps:
        sys.exit(0)

    epoch_cb = None
    train_dataiter = mx.io.PrefetchingIter(train_dataiter)

    model.fit(train_dataiter,
        begin_epoch        = begin_epoch,
        num_epoch          = 999999,
        eval_data          = val_dataiter,
        eval_metric        = eval_metrics,
        kvstore            = args.kvstore,
        optimizer          = opt,
        #optimizer_params   = optimizer_params,
        initializer        = initializer,
        arg_params         = arg_params,
        aux_params         = aux_params,
        allow_missing      = True,
        batch_end_callback = _batch_callback,
        epoch_end_callback = epoch_cb )


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 1. define global parameters
    args = get_parser()
    global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
    inc_op = tf.assign_add(global_step, 1, name='increment_global_step')
    images = tf.placeholder(name='img_inputs', shape=[None, *args.image_size, 3], dtype=tf.float32)
    labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
    # trainable = tf.placeholder(name='trainable_bn', dtype=tf.bool)
    dropout_rate = tf.placeholder(name='dropout_rate', dtype=tf.float32)
    # 2 prepare train datasets and test datasets by using tensorflow dataset api
    # 2.1 train datasets
    # the image is substracted 127.5 and multiplied 1/128.
    # random flip left right
    tfrecords_f = os.path.join(args.tfrecords_file_path, 'tran.tfrecords')
    dataset = tf.data.TFRecordDataset(tfrecords_f)
    dataset = dataset.map(parse_function)
    dataset = dataset.shuffle(buffer_size=args.buffer_size)
    dataset = dataset.batch(args.batch_size)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    # 2.2 prepare validate datasets
    ver_list = []
    ver_name_list = []
    for db in args.eval_datasets:
        print('begin db %s convert.' % db)
        data_set = load_bin(db, args.image_size, args)
        ver_list.append(data_set)
        ver_name_list.append(db)
    # 3. define network, loss, optimize method, learning rate schedule, summary writer, saver
    # 3.1 inference phase
    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
    net = get_resnet(images, args.net_depth, type='ir', w_init=w_init_method, trainable=True, keep_rate=dropout_rate)
    # 3.2 get arcface loss
    logit = arcface_loss(embedding=net.outputs, labels=labels, w_init=w_init_method, out_num=args.num_output)
    # test net  because of batch normal layer
    tl.layers.set_name_reuse(True)
    test_net = get_resnet(images, args.net_depth, type='ir', w_init=w_init_method, trainable=False, reuse=True, keep_rate=dropout_rate)
    embedding_tensor = test_net.outputs
    # 3.3 define the cross entropy
    inference_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels))
    # inference_loss_avg = tf.reduce_mean(inference_loss)
    # 3.4 define weight deacy losses
    # for var in tf.trainable_variables():
    #     print(var.name)
    # print('##########'*30)
    wd_loss = 0
    for weights in tl.layers.get_variables_with_name('W_conv2d', True, True):
        wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(weights)
    for W in tl.layers.get_variables_with_name('resnet_v1_50/E_DenseLayer/W', True, True):
        wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(W)
    for weights in tl.layers.get_variables_with_name('embedding_weights', True, True):
        wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(weights)
    for gamma in tl.layers.get_variables_with_name('gamma', True, True):
        wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(gamma)
    # for beta in tl.layers.get_variables_with_name('beta', True, True):
    #     wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(beta)
    for alphas in tl.layers.get_variables_with_name('alphas', True, True):
        wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(alphas)
    # for bias in tl.layers.get_variables_with_name('resnet_v1_50/E_DenseLayer/b', True, True):
    #     wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(bias)

    # 3.5 total losses
    total_loss = inference_loss + wd_loss
    # 3.6 define the learning rate schedule
    p = int(512.0/args.batch_size)
    lr_steps = [p*val for val in args.lr_steps]
    print(lr_steps)
    lr = tf.train.piecewise_constant(global_step, boundaries=lr_steps, values=[0.001, 0.0005, 0.0003, 0.0001], name='lr_schedule')
    # 3.7 define the optimize method
    opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=args.momentum)
    # 3.8 get train op
    grads = opt.compute_gradients(total_loss)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.apply_gradients(grads, global_step=global_step)
    # train_op = opt.minimize(total_loss, global_step=global_step)
    # 3.9 define the inference accuracy used during validate or test
    pred = tf.nn.softmax(logit)
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), labels), dtype=tf.float32))
    # 3.10 define sess
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=args.log_device_mapping)
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    # 3.11 summary writer
    summary = tf.summary.FileWriter(args.summary_path, sess.graph)
    summaries = []
    # # 3.11.1 add grad histogram op
    for grad, var in grads:
        if grad is not None:
            summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
    # 3.11.2 add trainabel variable gradients
    for var in tf.trainable_variables():
        summaries.append(tf.summary.histogram(var.op.name, var))
    # 3.11.3 add loss summary
    summaries.append(tf.summary.scalar('inference_loss', inference_loss))
    summaries.append(tf.summary.scalar('wd_loss', wd_loss))
    summaries.append(tf.summary.scalar('total_loss', total_loss))
    # 3.11.4 add learning rate
    summaries.append(tf.summary.scalar('leraning_rate', lr))
    summary_op = tf.summary.merge(summaries)
    # 3.12 saver
    saver = tf.train.Saver(max_to_keep=args.saver_maxkeep)
    # 3.13 init all variables
    sess.run(tf.global_variables_initializer())

    # restore_saver = tf.train.Saver()
    # restore_saver.restore(sess, '/home/aurora/workspaces2018/InsightFace_TF/output/ckpt/InsightFace_iter_1110000.ckpt')
    # 4 begin iteration
    if not os.path.exists(args.log_file_path):
        os.makedirs(args.log_file_path)
    log_file_path = args.log_file_path + '/train' + time.strftime('_%Y-%m-%d-%H-%M', time.localtime(time.time())) + '.log'
    log_file = open(log_file_path, 'w')
    # 4 begin iteration
    count = 0
    total_accuracy = {}

    for i in range(args.epoch):
        sess.run(iterator.initializer)
        while True:
            try:
                images_train, labels_train = sess.run(next_element)
                feed_dict = {images: images_train, labels: labels_train, dropout_rate: 0.4}
                feed_dict.update(net.all_drop)
                start = time.time()
                _, total_loss_val, inference_loss_val, wd_loss_val, _, acc_val = \
                    sess.run([train_op, total_loss, inference_loss, wd_loss, inc_op, acc],
                              feed_dict=feed_dict,
                              options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
                end = time.time()
                pre_sec = args.batch_size/(end - start)
                # print training information
                if count > 0 and count % args.show_info_interval == 0:
                    print('epoch %d, total_step %d, total loss is %.2f , inference loss is %.2f, weight deacy '
                          'loss is %.2f, training accuracy is %.6f, time %.3f samples/sec' %
                          (i, count, total_loss_val, inference_loss_val, wd_loss_val, acc_val, pre_sec))
                count += 1

                # save summary
                if count > 0 and count % args.summary_interval == 0:
                    feed_dict = {images: images_train, labels: labels_train, dropout_rate: 0.4}
                    feed_dict.update(net.all_drop)
                    summary_op_val = sess.run(summary_op, feed_dict=feed_dict)
                    summary.add_summary(summary_op_val, count)

                # save ckpt files
                if count > 0 and count % args.ckpt_interval == 0:
                    filename = 'InsightFace_iter_{:d}'.format(count) + '.ckpt'
                    filename = os.path.join(args.ckpt_path, filename)
                    saver.save(sess, filename)

                # validate
                if count > 0 and count % args.validate_interval == 0:
                    feed_dict_test ={dropout_rate: 1.0}
                    feed_dict_test.update(tl.utils.dict_to_one(net.all_drop))
                    results = ver_test(ver_list=ver_list, ver_name_list=ver_name_list, nbatch=count, sess=sess,
                             embedding_tensor=embedding_tensor, batch_size=args.batch_size, feed_dict=feed_dict_test,
                             input_placeholder=images)
                    print('test accuracy is: ', str(results[0]))
                    total_accuracy[str(count)] = results[0]
                    log_file.write('########'*10+'\n')
                    log_file.write(','.join(list(total_accuracy.keys())) + '\n')
                    log_file.write(','.join([str(val) for val in list(total_accuracy.values())])+'\n')
                    log_file.flush()
                    if max(results) > 0.996:
                        print('best accuracy is %.5f' % max(results))
                        filename = 'InsightFace_iter_best_{:d}'.format(count) + '.ckpt'
                        filename = os.path.join(args.ckpt_path, filename)
                        saver.save(sess, filename)
                        log_file.write('######Best Accuracy######'+'\n')
                        log_file.write(str(max(results))+'\n')
                        log_file.write(filename+'\n')

                        log_file.flush()
            except tf.errors.OutOfRangeError:
                print("End of epoch %d" % i)
                break
    log_file.close()
    log_file.write('\n')