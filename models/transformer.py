from typing import Tuple, Optional, List
import torch
import torch.nn as nn
import copy

from misc.utils import pack_wrapper, clip_att, subsequent_mask
from .caption_model import CaptionModel
from .layers import *

class TransformerModel(CaptionModel):
    """
    Vanilla transformer captioning model with spatial positional encoding
    """
    def __init__(self,
                 vocab_size: int,
                 num_layers: int,
                 input_encoding_size: int,
                 seq_length: int,
                 img_feat_size: int,
                 ff_size: int,
                 heads: int,
                 cross_attention: Callable[[int, int, float], nn.Module],
                 norm: Callable[[int], nn.Module],
                 ff_activation: nn.Module,
                 dropout: float,
                 enc_pos_embedding: Optional[Callable[[int, float], nn.Module]],
                 ):

        super().__init__()
        
        self.vocab_size = vocab_size
        # self.input_encoding_size = input_encoding_size
        self.seq_length = seq_length
        # self.img_feat_size = img_feat_size
        self.ss_prob = 0.0 # Schedule sampling probability

        self.att_embed = nn.Sequential(nn.Linear(img_feat_size, input_encoding_size),
                                       nn.ReLU(),
                                       nn.Dropout(dropout))

        self.model = self.make_model(self.vocab_size + 1,
                                     num_layers,
                                     input_encoding_size,
                                     ff_size,
                                     heads,
                                     enc_pos_embedding,
                                     cross_attention,
                                     norm,
                                     ff_activation,
                                     dropout)
    
    @staticmethod
    def make_model(tgt_vocab: int,
                   N: int,# = 6,
                   d_model: int,# = 512,
                   d_ff: int,# = 2048,
                   heads: int,# = 8,
                   enc_pos_embedding: Optional[Callable[[int, float], nn.Module]],
                   cross_attention: Callable[[int, int, float], nn.Module],
                   norm: Callable[[int], nn.Module],
                   ff_activation: nn.Module,
                   dropout: float = 0.1):
        """
        Constructs the model from hyperparameters with Xavier initialization.

        Parameters
        ----------
        src_vocab : int
                    Size of source vocabulary
        tgt_vocab : int
                    Size of target vocabulary
        N         : int, optional
                    Number of layers. Default is 6.
        d_model   : int, optional
                    Input feature size. Default is 512.
        d_ff      : int, optional
                    Intermediate feature size of Position wise Feed Forward N/W.
                    Default is 2048.
        heads     : int, optional
                    Number of attention heads. Default is 8.
        dropout   : float, optional
                    Dropout probability. Default is 0.1.

        Returns
        -------
        model     : EncoderDecoder
                    Model constructed with the given hyperparameters and Xavier
                    initialized.
        """
        c = copy.deepcopy
        # CrossAttention = {'xlinear' : XLinearMultiHeadedAttention, 'dot-product' : MultiHeadedAttention}
        self_attn = MultiHeadedAttention(heads, d_model, dropout)
        cross_attention = cross_attention(1 if isinstance(cross_attention(1,10,0.5), XLinearMultiHeadedAttention) else heads, d_model, dropout)
        # cross_attention = CrossAttention[cross_attention](heads if cross_attention == 'dot-product' else 1, d_model, dropout = dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout, c(ff_activation))
        enc_position = enc_pos_embedding(d_model, dropout) if enc_pos_embedding else lambda x, y: x
        # enc_position = SpatialPositionalEncoding(grids, d_model, dropout, enc_learnable_pos, enc_learnable_pos_type) if use_grid else lambda x,y : x
        dec_position = PositionalEncoding(d_model, dropout)
        model = EncoderDecoder(Encoder(EncoderLayer(d_model, c(self_attn), c(ff), norm, dropout), c(norm), N),
                               Decoder(DecoderLayer(d_model, c(self_attn), c(cross_attention), c(ff), c(norm), dropout), c(norm), N),
                               enc_position,
                               nn.Sequential(Embeddings(d_model, tgt_vocab), c(dec_position)),
                               Generator(d_model, tgt_vocab))
        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        return model

    def logit(self, x): # unsafe way
        return self.model.generator.proj(x)

    def _forward(self,
                 image_features: torch.Tensor,
                 seq: torch.Tensor,
                 image_masks: Optional[torch.Tensor] = None,
                 boxes: Optional[torch.Tensor] = None
                 ) -> torch.Tensor:
        """
        Parameters
        ----------
        image_features : torch.tensor of shape (B, L, D)
                    Output of last conv layer of CNN or bottom-up features
        seq       : torch.tensor of shape (B, T+2)
                    1-indexed captions including <START> and <END>
        image_masks : torch.tensor of shape (B, L) or None
                    Attention mask when no. of bottom-up proposals are
                    unequal across batch.

        Returns
        -------
        outputs   : torch.tensor of shape (B, T+1, V)
                    Distribution over vocabulary of the output sequence.
        """
        # Compute the visual embedding, remove end token and get causal sequence mask
        image_features, seq, boxes, image_masks, seq_mask = self._prepare_feature(image_features, boxes, image_masks, seq)
        # Pass all the features through transformer encoder-decoder, (B,T+1,H)
        out = self.model(image_features, seq, boxes, image_masks, seq_mask)
        # Project output to vocabulary size and compute log_softmax, (B,T+1,H)->(B,T+1,V)
        outputs = self.model.generator(out)
        return outputs

    def _prepare_feature(self,
                         image_features: torch.Tensor,
                         boxes: Optional[torch.Tensor],
                         image_masks: Optional[torch.Tensor] = None,
                         seq: Optional[torch.Tensor] = None
                         ) -> Tuple[torch.Tensor,
                                    Optional[torch.Tensor],
                                    Optional[torch.Tensor],
                                    torch.Tensor,
                                    Optional[torch.Tensor]]:
        """
        Computes the embeddings of visual features, removes end token and
        prepares causal sequence mask.

        Parameters
        ----------
        image_features : torch.tensor of shape (B, L, D)
                    Output of last conv layer of CNN or bottom-up features
        image_masks : torch.tensor of shape (B, L), optional
                    Attention mask when no. of bottom-up proposals are unequal
                    across batch. Default is None.
        seq       : torch.tensor of shape (B, T+2), optional
                    1-indexed captions including <START> and <END>. Default is
                    None.

        Returns
        -------
        image_features : torch.tensor of shape (B, P, E)
                    image_features after passing through attention embedding layers.
        seq       : torch.tensor of shape (B, T+1)
                    1-indexed captions including <START> but excluding <END>.
        image_masks : torch.tensor of shape (B, 1, P)
                    Attention mask.
        seq_mask  : torch.tensor of shape (B, T+1, T+1)
                    Causal sequence mask.
        """
        # Clip to maximum feature length. Required when using multi-GPU. (B,L,D)->(B,P,D)
        image_features, boxes, image_masks = clip_att(image_features, boxes, image_masks)
        # Applies self.att_embed layer on image_features, (B,P,D)->(B,P,E)
        image_features = pack_wrapper(self.att_embed, image_features, image_masks)
        # If no attention mask, create a mask with all ones, (B,P)
        if image_masks is None:
            image_masks = image_features.new_ones(image_features.shape[:2], dtype=torch.long)
        # (B,P)->(B,1,P)
        image_masks = image_masks.unsqueeze(-2)
        # If sequence is available as input
        if seq is not None:
            # Crop the last token. Not to be decoded. (B,T+2)->(B,T+1)
            seq = seq[:,:-1]
            # Create sequence mask
            seq_mask = (seq > 0)
            # Unmask <START> token
            seq_mask[:,0] += True
            # (B,T+1)->(B,1,T+1)
            seq_mask = seq_mask.unsqueeze(-2)
            # Create causal sequence mask, (B,1,T+1)->(B,T+1,T+1)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return image_features, seq, boxes, image_masks, seq_mask

    def get_logprobs_state(self,
                           it: torch.Tensor,
                           enc_out: torch.Tensor,
                           mask: torch.Tensor,
                           state: torch.Tensor
                           ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Finds the log probability distribution of the next word given the encoder
        output and all previous words.

        Parameters
        ----------
        it        : torch.tensor of shape (B,)
                    Current word
        enc_out   : torch.tensor of shape (B, P, E)
                    Encoder output
        mask      : torch.tensor of shape (B, P)
                    Mask for encoder output
        state     : list of length 1 or None
                    torch.tensor of shape (1, B, t) containing indices of words
                    generated upto time step t-1 including <START>.

        Returns
        -------
        log_probs : torch.tensor of shape (B, V)
                    Log softmax distribution over vocabulary for t+1
        state     : list of length 1 containing torch.tensor of shape (1, B, t+1)
                    Words generated upto time step t including <START>.
        """
        if state is None:
            ys = it.unsqueeze(1) #(B,1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim = 1) # (B,t+1)
        # Compute target sequence embedding and run the decoder, (B,t+1,H), (B,2,H)
        dec_out = self.model.decode(enc_out, mask, ys, subsequent_mask(ys.size(1)).to(enc_out.device))
        # Compute output distribution for the last time step, (B, V)
        logprobs = self.model.generator(dec_out[:, -1])

        return logprobs, [ys.unsqueeze(0)] # (B,V), (1,B,t+1)

    def _sample_beam(self,
                     image_features: torch.Tensor,
                     boxes: Optional[torch.Tensor],
                     image_masks: Optional[torch.Tensor] = None,
                     beam_width: int = 10,
                     group_size: int = 1,
                     decoding_constraint: bool = False,
                     diversity_lambda: float = 0.5,
                     perplexity: bool = False
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate captions for the given input using beam search.

        Args:
            image_features (torch.Tensor): Image features (B,L,D)
            boxes (Optional[torch.Tensor]): Bounding box features
            image_masks (Optional[torch.Tensor], optional): Image feature mask when no. of features are unequal
            across batch (B,L). Defaults to None.
            beam_width (int, optional): Width of beam search. Defaults to 10.
            group_size (int, optional): Group size for diverse beam search. Defaults to 1.
            decoding_constraint (bool, optional): Whether not to allow same words in a row. Defaults to False.
            diversity_lambda (float, optional): Lambda for diverse beam search. Defaults to 0.5.
            perplexity (bool, optional): Whether to use perplexity instead of probability. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sampled sequence (B,T), Log probability of the samples (B,T)
        """
        batch_size = image_features.size(0)
        # Compute the visual embedding, seq = None, seq_mask = None
        image_features, seq, boxes, image_masks, seq_mask = self._prepare_feature(image_features, boxes, image_masks)
        # Encode the visual features, (B,P,E)
        memory = self.model.encode(image_features, boxes, image_masks)

        assert beam_width <= self.vocab_size + 1, 'Lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)

        # Lets process every image independently for now, for simplicity
        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = None
            # Get kth visual feature and replicate beam_width times
            tmp_memory = memory[k:k+1].expand(*((beam_width,)+memory.size()[1:])).contiguous()
            # Get kth attention mask and replicate beam_width times
            tmp_image_masks = image_masks[k:k+1].expand(*((beam_width,)+image_masks.size()[1:])).contiguous() if image_masks is not None else None
            # Initial input = <START>
            it = memory.new_zeros([beam_width], dtype=torch.long)
            # Get the log probability of the next word, (beam_width, V), list-(1, beam_width, 1)
            logprobs, state = self.get_logprobs_state(it, tmp_memory, tmp_image_masks, state)
            # Perform beam serach
            self.done_beams[k] = self.beam_search(state, logprobs, tmp_memory, tmp_image_masks, beam_width, group_size, diversity_lambda, decoding_constraint, perplexity)
            # The first beam has highest cumulative score
            seq[:, k] = self.done_beams[k][0]['seq']
            # Get the sequence log probabilities of the first beam
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # Return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self,
                image_features: torch.Tensor,
                image_masks: Optional[torch.Tensor],# = None,
                boxes: Optional[torch.Tensor],# = None,
                sample_max: bool,# = True,
                beam_width: int,# = 1,
                group_size: int,
                diversity_lambda: float,
                perplexity: bool,
                temperature: float = 1.0,
                decoding_constraint: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate captions for the given input by greedy decoding, sampling or beam search.

        Args:
            image_features (torch.Tensor): Image feature (B,L,D)
            image_masks (Optional[torch.Tensor], optional): Image feature mask when no. of features
            are unequal across batch (B,L). Defaults to None.
            boxes (Optional[torch.Tensor], optional): _description_. Defaults to None.
            sample_max (bool, optional): Whether to do greedy decoding. Defaults to True.
            beam_width (int, optional): Width of beam search. Defaults to 1.
            temperature (float, optional): Sampling temperature. Defaults to 1.0.
            decoding_constraint (bool, optional): Whether not to allow same words in a row. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sampled sequence (B,T), Log probability of the samples (B,T)
        """
        if beam_width > 1:
            return self._sample_beam(image_features, boxes, image_masks, beam_width, group_size, decoding_constraint, diversity_lambda, perplexity)

        batch_size = image_features.shape[0]
        # Compute the visual embedding, (B,P,E), seq = None, (B,P), seq_mask = None
        image_features, seq, boxes, image_masks, seq_mask = self._prepare_feature(image_features, boxes, image_masks)

        state = None
        # Encode the visual features, (B,P,E)
        memory = self.model.encode(image_features, boxes, image_masks)
        # Tensor for storing sequence and log probabilities
        seq = image_features.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = image_features.new_zeros(batch_size, self.seq_length)

        for t in range(self.seq_length + 1):
            if t == 0: # input <START>
                it = memory.new_zeros(batch_size, dtype=torch.long)
            # Get the log probability of the next word, (B,V), [(1,B,t+1)]
            logprobs, state = self.get_logprobs_state(it, memory, image_masks, state)
            # Whether not to allow same word in a row
            if decoding_constraint and t > 0:
                tmp = seqLogprobs.new_zeros(seqLogprobs.shape[0], self.vocab_size + 1)
                tmp.scatter_(1, seq[:,t-1].unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp
            # Skip if we achieve maximum length
            if t == self.seq_length:
                break
            # Perform greedy decoding
            if sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            # Perform temperature scaling if required
            else:
                if temperature == 1.0: # Fetch prev distribution, (B,V)
                    prob_prev = torch.exp(logprobs.data)
                else: # Scale logprobs by temperature SHOULDN'T THE PROBABILITY BE RE-NORMALIZED?
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1) #(B,V)->(B,1)
                # Gather the logprobs at sampled positions (B,V)->(B,1)
                sampleLogprobs = logprobs.gather(1, it)
                it = it.view(-1).long() # Flatten indices for downstream processing (B,)

            # Find sequences which have not generated <END> so far
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:,t] = it
            seqLogprobs[:,t] = sampleLogprobs.view(-1)
            # Quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs
