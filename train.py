from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
# torch.manual_seed(0)

import numpy as np

import time
import os
from six.moves import cPickle
from datetime import datetime

import opts
import models
from dataloader import DataLoader
import eval_utils
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward

try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def train(opt):
    # Initialize tensorboard
    tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)
    # Create checkpoint path
    if not os.path.isdir(opt.checkpoint_path):
        os.makedirs(opt.checkpoint_path)

    infos = {}
    histories = {}
    # Load histories and infos for resuming training
    if opt.start_from is not None:
        # Open old infos and check if models are compatible
        infos = utils.load(os.path.join(opt.start_from, 'infos.pkl'))
        saved_model_opt = infos['opt']
        need_be_same = ["caption_model", "ff_size", "num_layers"]
        for checkme in need_be_same:
            assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '{:s}' ".format(checkme)
        if os.path.isfile(os.path.join(opt.start_from, 'histories.pkl')):
            histories = utils.load(os.path.join(opt.start_from, 'histories.pkl'))

    train_loader = DataLoader(opt, 'train', infos.get('train', {}))
    val_loader = DataLoader(opt, 'val', length = opt.val_images_use)

    # Get current iteration/step if available
    iteration = infos.get('iter', 0)
    # Get current epoch if available
    start_epoch = infos.get('epoch', 0)
    # Get validation result history if available
    val_result_history = histories.get('val_result_history', {})
    # Get loss history if available
    loss_history = histories.get('loss_history', {})
    # Get lr history if available
    lr_history = histories.get('lr_history', {})
    # Get ss probability if available
    ss_prob_history = histories.get('ss_prob_history', {})
    # Get size of vocabulary
    opt.vocab_size = train_loader.vocab_size()
    # Get max sequence length
    opt.seq_length = train_loader.seq_length()

    # If enabled, load best score so far
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if opt.device == "cuda" else torch.device("cpu")

    # Create model
    model = models.setup(opt, device)
    # Ensure in training mode
    model.train()

    # Get loss functions
    if opt.label_smoothing > 0: # Label smoothing + KL Divergence loss
        crit = utils.LabelSmoothing(vocab_size = opt.vocab_size, smoothing = opt.label_smoothing)
    else: # Cross entropy loss
        crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion() # Self critical loss

    if opt.noamopt:
        optimizer = utils.get_std_opt(model, factor = opt.noamopt_factor, warmup = opt.noamopt_warmup)
        optimizer._step = iteration
    elif opt.reduce_on_plateau:
        optimizer = utils.build_optimizer(model.parameters(), opt)
        optimizer = utils.ReduceLROnPlateau(optimizer, factor = 0.5, patience = 3)
    else:
        optimizer = utils.build_optimizer(model.parameters(), opt)
    # Load the optimizer
    if vars(opt).get('start_from', None) is not None and os.path.isfile(os.path.join(opt.start_from, "optimizer.pt")):
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pt'), map_location = device))    

    # Beginning of training loop
    for epoch in range(start_epoch, opt.max_epochs):
        # Decay learning rate
        if not opt.noamopt and not opt.reduce_on_plateau:
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate  ** frac
                opt.current_lr = opt.learning_rate * decay_factor
            else:
                opt.current_lr = opt.learning_rate
            utils.set_lr(optimizer, opt.current_lr) # set the decayed rate

        # Assign scheduled sampling prob
        if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start > 0:
            frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
            opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
            model.ss_prob = opt.ss_prob

        # If starting self critical training
        if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
            sc_flag = True
            init_scorer(opt.cached_tokens)
        else:
            # utils.change_local_alpha(model, 2, epoch)
            sc_flag = False

        start = time.time()
        # Process each mini batch of data
        for data in train_loader:
            # Move the data to device if data is not None
            tmp = [data['img_feats'], data['labels'], data['label_masks'], data['img_masks'], data['boxes']]
            tmp = [x if x is None else x.to(device) for x in tmp]
            img_feats, labels, label_masks, img_masks, boxes = tmp
            # Reset gradients
            optimizer.zero_grad()
            # If not using self critical training
            if not sc_flag:
                # Perform forward pass, (B,T+1,V)
                seqLogprobs = model(img_feats, labels, img_masks, boxes)
                # Compute loss, ([])
                loss = crit(seqLogprobs, labels[:,1:], label_masks[:,1:])
                # Compute gradients
                loss.backward()
                train_loss = loss.item()
            # If using self critical training, perform sampling -> compute reward
            else: 
                # Perform sampling, (B,T), (B,T)
                gen_result, sample_logprobs = model(img_feats, img_masks, boxes, opt = {'sample_max':0}, mode = 'sample')
                # Get self critical reward, (B,T)
                reward = get_self_critical_reward(model, img_feats, img_masks, boxes, data['gts'], gen_result, opt)
                # Compute loss, ([])
                loss = rl_crit(sample_logprobs, gen_result.data, reward)
                # Compute gradients
                loss.backward()
                train_loss = loss.item()

            # Clip the gradients
            utils.clip_gradient(optimizer, opt.grad_clip)
            # Update the parameters
            optimizer.step()

            end = time.time()

            # Update the iteration and epoch
            iteration += 1

            # Print results for the current iteration
            if not sc_flag:
                print("Iter {} (epoch {}), loss = {:.3f}, time/batch = {:.3f}".format(iteration, epoch, train_loss, end - start))
            else:
                print("Iter {} (epoch {}), avg reward = {:.3f}, time/batch = {:.3f}".format(iteration, epoch, reward[:,0].mean().item(), end - start))

            # Write the training loss summary
            if (iteration % opt.losses_log_every == 0):
                add_summary_value(tb_summary_writer, 'train_loss', train_loss, iteration)
                if opt.noamopt:
                    opt.current_lr = optimizer.rate()
                elif opt.reduce_on_plateau:
                    opt.current_lr = optimizer.current_lr
                add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)
                add_summary_value(tb_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
                if sc_flag:
                    add_summary_value(tb_summary_writer, 'avg_reward', reward[:,0].mean().item(), iteration)

                loss_history[iteration] = train_loss if not sc_flag else reward[:,0].mean().item()
                lr_history[iteration] = opt.current_lr
                ss_prob_history[iteration] = model.ss_prob

            # Make evaluation on validation set, and save model
            if (iteration % opt.perform_validation_every == 0):
                # Eval parameters
                eval_kwargs = {'split': 'val',
                               'dataset': opt.input_json,
                               'use_box': opt.use_box}
                eval_kwargs.update(vars(opt))
                # Evaluate the model on the validation split
                val_loss, predictions, lang_stats = eval_utils.eval_split(model, crit, val_loader, eval_kwargs, device)
                # If using ReduceLROnPlateau, run a scheduling step
                if opt.reduce_on_plateau:
                    # Use CIDEr if available as the monitoring quantity
                    if 'CIDEr' in lang_stats:
                        optimizer.scheduler_step(-lang_stats['CIDEr'])
                    # Use val loss as the monitoring quantity
                    else:
                        optimizer.scheduler_step(val_loss)

                # Write validation result into summary
                add_summary_value(tb_summary_writer, 'validation loss', val_loss, iteration)
                if lang_stats:
                    for k,v in lang_stats.items():
                        add_summary_value(tb_summary_writer, k, v, iteration)
                val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

                # Save model if it is improving on validation result
                if opt.language_eval == 1:
                    current_score = lang_stats['CIDEr']
                else:
                    current_score = -val_loss

                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pt')
                    torch.save(model.state_dict(), checkpoint_path)
                    utils.save(infos, os.path.join(opt.checkpoint_path, 'infos-best.pkl'))
                    print("Best model saved to {}".format(checkpoint_path))


            if (iteration % opt.save_checkpoint_every) == 0:
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pt')
                torch.save(model.state_dict(), checkpoint_path)
                optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pt')
                torch.save(optimizer.state_dict(), optimizer_path)
                # Dump miscalleous informations
                infos = {'train'          : train_loader.state_dict(),
                         'iter'           : iteration,
                         'epoch'          : epoch,
                         'best_val_score' : best_val_score,
                         'opt'            : opt,
                         'vocab'          : train_loader.get_vocab()}

                histories = {'val_result_history' : val_result_history,
                             'loss_history'       : loss_history,
                             'lr_history'         : lr_history,
                             'ss_prob_history'    : ss_prob_history}

                utils.save(infos, os.path.join(opt.checkpoint_path, 'infos.pkl'))
                utils.save(histories, os.path.join(opt.checkpoint_path, 'histories.pkl'))
                print("Model saved to {}".format(checkpoint_path))
            
            start = time.time()

if __name__ == "__main__":
    opt = opts.parse_opt()
    train(opt)