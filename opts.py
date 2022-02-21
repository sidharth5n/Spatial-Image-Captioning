import argparse

def parse_opt():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--input_json', type=str, default='/data/captioning_data/dataset_coco.json',
                    help='path to the json file containing additional info and vocab')
    parser.add_argument('--input_fc_feat_dir', type=str, default='data/cocobu_fc',
                    help='path to the directory containing the preprocessed fc feats')
    parser.add_argument('--input_img_feat_dir', type=str, default='data/cocobu_att',
                    help='path to the directory containing the preprocessed att feats')
    parser.add_argument('--input_box_dir', type=str, default='data/cocobu_box',
                    help='path to the directory containing the boxes of att feats')
    parser.add_argument('--input_rel_box_dir',type=str, default='/data/cocobu_rel_box',
                    help="this directory contains the bboxes in relative coordinates for the corresponding image features in --input_att_dir")
    parser.add_argument('--input_grid_vec_dir',type=str, default='/data/cocobu_box_vec',
                    help="this directory contains the grid features of the relative coordinate bboxes for the corresponding image features in --input_img_feat_dir")
    parser.add_argument('--input_label_h5', type=str, default='data/coco_label.h5',
                    help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--start_from', type=str, default=None,
                    help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                        'infos.pkl'         : configuration;
                        'checkpoint'        : paths to model file(s)
                                              Note: this file contains absolute paths, be careful when moving files around;
                        'model.ckpt-*'      : file(s) with model definition
                    """)
    parser.add_argument('--cached_tokens', type=str, default='coco-train-idxs',
                    help='Cached token file for calculating cider score during self critical training.')

    # Model settings
    parser.add_argument('--caption_model', type=str, default="vanilla",
                    help='ort, local, spatial, new_local, vanilla, spatial_pe, show_tell')
    parser.add_argument('--ff_size', type=int, default=512,
                    help='Size of first layer in position wise feed forward network')
    parser.add_argument('--ff_activation', type=str, default='RELU', choices = ['RELU', 'GELU'],
                    help="Activation function to be used in position wise feed forward network. One of RELU, GELU.")
    parser.add_argument('--norm', type=str, default='layer', choices = ['layer', 'RMS'],
                    help="Normalization type. One of layer, RMS.")
    parser.add_argument('--num_layers', type=int, default=1,
                    help='Number of layers to stack')
    parser.add_argument('--rnn_type', type=str, default='lstm',
                    help='Type of RNN for RNN based models')
    parser.add_argument('--hidden_size', type=int, default=512,
                    help='Number of hidden nodes in each layer of RNN')
    parser.add_argument('--num_grids', type=int, default=576,
                    help='Number of grids in which the image is divided for box vector creation')
    parser.add_argument('--input_encoding_size', type=int, default=512,
                    help='The encoding size of each token in the vocabulary, and the image.')
    parser.add_argument('--img_feat_size', type=int, default=2048,
                    help='2048 for resnet, 512 for vgg')
    parser.add_argument('--heads', type=int, default=8,
                    help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.5,
                    help='Strength of dropout')
    parser.add_argument('--use_grid', type=str2bool, default=False,
                    help='Whether to use box grid vectors')
    parser.add_argument('--enc_pos_embedding', type=str2bool, default=True,
                    help='Whether to use learnable position embedding in encoder when using grid vectors')
    parser.add_argument('--enc_pos_type', type = str, default = 'gxg', choices = ['gxg', 'g2', 'box-coord'],
                    help = 'Type of encoder positional encoding if using learnt encoding')
    parser.add_argument('--use_box', type=str2bool, default=False,
                    help='Whether to use box features')
    parser.add_argument('--use_local_attn', type=str2bool, default=False,
                    help='Whether to use local attention in decoder')
    parser.add_argument('--local_win_size', type=int, default=3,
                    help="Size of window to be used in local attention")
    parser.add_argument('--adaptive_dec_self_attn', type=str2bool, default=False,
                    help="Whether to use adaptive combination of causal and local self attention")
    parser.add_argument('--cross_attn', type=str, default='xlinear',
                    help="Type of cross attention to be used in decoder. One of xlinear or dot-product")

    # feature manipulation
    parser.add_argument('--norm_img_feat', type=int, default=0,
                    help='If normalize attention features')
    parser.add_argument('--norm_box_feat', type=int, default=0,
                    help='If use box, do we normalize box feature')

    # Optimization: General
    parser.add_argument('--max_epochs', type=int, default=30,
                    help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                    help='minibatch size')
    parser.add_argument('--grad_clip', type=float, default=0.1, #5.,
                    help='clip gradients at this value')
    parser.add_argument('--self_critical_after', type=int, default=-1,
                    help='After what epoch do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
    parser.add_argument('--seq_per_img', type=int, default=5,
                    help='number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')
    parser.add_argument('--beam_size', type=int, default=1,
                    help='Used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')

    # Optimization: for the Language Model
    parser.add_argument('--optim', type=str, default='adam',
                    help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--learning_rate', type=float, default=4e-4,
                    help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=-1,
                    help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=3,
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8,
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--optim_alpha', type=float, default=0.9,
                    help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                    help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                    help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight_decay')

    # Scheduled sampling
    parser.add_argument('--scheduled_sampling_start', type=int, default=-1,
                    help='at what iteration to start decay gt probability')
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5,
                    help='every how many epochs thereafter to gt probability')
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05,
                    help='How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25,
                    help='Maximum scheduled sampling prob.')


    # Evaluation/Checkpointing
    parser.add_argument('--val_images_use', type=int, default=3200,
                    help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--save_checkpoint_every', type=int, default=100,
                    help='how often to save a model checkpoint (in iterations)?')
    parser.add_argument('--perform_validation_every', type=int, default=2500,
                    help='how often to perform a validation (in iterations)?')
    parser.add_argument('--checkpoint_path', type=str, default='save',
                    help='directory to store checkpointed models')
    parser.add_argument('--language_eval', type=int, default=0,
                    help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
    parser.add_argument('--losses_log_every', type=int, default=15,
                    help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
    parser.add_argument('--load_best_score', type=int, default=1,
                    help='Do we load previous best score when resuming training.')

    # misc
    parser.add_argument('--id', type=str, default='',
                    help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    parser.add_argument('--train_only', type=str2bool, default=False,
                    help='if true then use 80k, else use 110k')
    parser.add_argument('--device', type=str, default='cuda',
                    help="Device to be used. One of 'cuda' or 'cpu'.")

    # Reward
    parser.add_argument('--cider_reward_weight', type=float, default=1,
                    help='The reward weight from cider')
    parser.add_argument('--bleu_reward_weight', type=float, default=0,
                    help='The reward weight from bleu4')

    # Transformer
    parser.add_argument('--label_smoothing', type=float, default=0,
                    help='')
    parser.add_argument('--noamopt', action='store_true',
                    help='')
    parser.add_argument('--noamopt_warmup', type=int, default=20000,
                    help='')
    parser.add_argument('--noamopt_factor', type=float, default=1,
                    help='')

    parser.add_argument('--reduce_on_plateau', action='store_true',
                    help='')

    #Relative geometry
    parser.add_argument("--box_trignometric_embedding", type=str2bool,
                        default=True)

    args = parser.parse_args()

    # Check if args are valid
    assert args.ff_size > 0, "rnn_size should be greater than 0"
    assert args.num_layers > 0, "num_layers should be greater than 0"
    assert args.input_encoding_size > 0, "input_encoding_size should be greater than 0"
    assert args.batch_size > 0, "batch_size should be greater than 0"
    assert args.dropout >= 0 and args.dropout < 1, "drop_prob_lm should be between 0 and 1"
    assert args.seq_per_img > 0, "seq_per_img should be greater than 0"
    assert args.beam_size > 0, "beam_size should be greater than 0"
    assert args.save_checkpoint_every > 0, "save_checkpoint_every should be greater than 0"
    assert args.losses_log_every > 0, "losses_log_every should be greater than 0"
    assert args.language_eval == 0 or args.language_eval == 1, "language_eval should be 0 or 1"
    assert args.load_best_score == 0 or args.load_best_score == 1, "language_eval should be 0 or 1"
    assert args.train_only == 0 or args.train_only == 1, "language_eval should be 0 or 1"
    assert (args.use_grid and args.use_box) == 0, "Only one of use_grid and use_box should be True"
    assert args.device in ['cpu', 'cuda'], "device has to be on of 'cpu' or 'cuda'"
    if args.noamopt:
        assert args.caption_model in ['local', 'ort', 'spatial', 'diff_local', 'new_local', 'diff_local2', 'vanilla', 'spatial_pe', 'spatial_pe2', 'spatial_pe3', 'normed_spatial_pe', 'normed_spatial_pe2'], 'noamopt can only work with transformer'
    elif args.caption_model == 'show_tell':
        assert args.input_fc_feat_dir is not None
    if args.caption_model == 'spatial':
        assert args.use_box, "use_box must be True when using spatial transformer"

    return args


def str2bool(v):
    if v.lower() in ('true', '1'):
        return True
    elif v.lower() in ('false', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
