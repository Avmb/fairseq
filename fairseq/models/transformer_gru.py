# Transformer encoder with GRU decoder

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import (
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import TransformerEncoder
from fairseq.modules import (
    LayerNorm, 
    MultiheadAttention,
    FairseqDropout)

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1e5

class GRUDecoder(FairseqIncrementalDecoder):
    """
    GRU decoder
    
    """
    
    def __init__(
        self,
        args,
        dictionary,
        embed_dim=512,
        hidden_size=512,
        out_embed_dim=512,
        num_layers=1,
        dropout_in=0.1,
        dropout_out=0.1,
        attention=True,
        encoder_output_units=512,
        pretrained_embed=None,
        share_input_output_embed=False,
        adaptive_softmax_cutoff=None,
        max_target_positions=DEFAULT_MAX_TARGET_POSITIONS,
        residuals=False,
    ):
        super().__init__(dictionary)
        self.args = args
        self.dropout_in_module = FairseqDropout(
            dropout_in, module_name=self.__class__.__name__
        )
        self.dropout_out_module = FairseqDropout(
            dropout_out, module_name=self.__class__.__name__
        )
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed
        self.need_attn = True
        self.max_target_positions = max_target_positions
        self.residuals = residuals
        self.num_layers = num_layers

        self.adaptive_softmax = None
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.encoder_output_units = encoder_output_units

        # disable input feeding if there is no encoder
        # input feeding is described in arxiv.org/abs/1508.04025
        input_feed_size = 0 if encoder_output_units == 0 else hidden_size
        self.layers = nn.ModuleList(
            [
                GRUCell(
                    input_size=input_feed_size + embed_dim
                    if layer == 0
                    else hidden_size,
                    hidden_size=hidden_size,
                )
                for layer in range(num_layers)
            ]
        )

        if attention:
            # TODO make bias configurable
            #self.attention = AttentionLayer(
            #    hidden_size, encoder_output_units, hidden_size, bias=False
            #)
            self.attention = MultiheadAttention(
                embed_dim,
                args.decoder_attention_heads,
                kdim=encoder_output_units,
                vdim=encoder_output_units,
                dropout=args.attention_dropout,
                encoder_decoder_attention=True,)
        else:
            self.attention = None
            
        if args.decoder_hidden_state_in_output:
            self.hidden_attn_combination_fc = Linear(2*hidden_size, hidden_size, bias=False)

        if hidden_size != out_embed_dim:
            self.additional_fc = Linear(hidden_size, out_embed_dim)

        if adaptive_softmax_cutoff is not None:
            # setting adaptive_softmax dropout to dropout_out for now but can be redefined
            self.adaptive_softmax = AdaptiveSoftmax(
                num_embeddings,
                hidden_size,
                adaptive_softmax_cutoff,
                dropout=dropout_out,
            )
        elif not self.share_input_output_embed:
            #self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)
            self.fc_out = Linear(out_embed_dim, num_embeddings)

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        src_lengths: Optional[Tensor] = None,
    ):
        x, attn_scores = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state
        )
        return self.output_layer(x), attn_scores

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        """
        Similar to *forward* but only return features.
        """
        # get outputs from encoder
        if encoder_out is not None:
            encoder_outs = encoder_out['encoder_out'][0]
            #encoder_hiddens = encoder_out[1]
            encoder_padding_mask = encoder_out['encoder_padding_mask'][0]
        else:
            encoder_outs = torch.empty(0)
            #encoder_hiddens = torch.empty(0)
            encoder_padding_mask = torch.empty(0)
        srclen = encoder_outs.size(0)

        if incremental_state is not None and len(incremental_state) > 0:
            prev_output_tokens = prev_output_tokens[:, -1:]

        bsz, seqlen = prev_output_tokens.size()

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = self.dropout_in_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        if incremental_state is not None and len(incremental_state) > 0:
            prev_hiddens, input_feed = self.get_cached_state(
                incremental_state
            )
        elif encoder_out is not None:
            # setup recurrent cells
            zero_state = x.new_zeros(bsz, self.hidden_size)
            prev_hiddens = [zero_state for i in range(self.num_layers)]
            input_feed = x.new_zeros(bsz, self.hidden_size)
        else:
            # setup zero cells, since there is no encoder
            zero_state = x.new_zeros(bsz, self.hidden_size)
            prev_hiddens = [zero_state for i in range(self.num_layers)]
            input_feed = None

        assert (
            srclen > 0 or self.attention is None
        ), "attention is not supported if there are no encoder outputs"
        attn_scores: Optional[Tensor] = (
            x.new_zeros(srclen, seqlen, bsz) if self.attention is not None else None
        )
        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            if input_feed is not None:
                input = torch.cat((x[j, :, :], input_feed), dim=1)
            else:
                input = x[j]

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden = rnn(input, prev_hiddens[i])

                # hidden state becomes the input to the next layer
                input = self.dropout_out_module(hidden)
                if self.residuals:
                    input = input + prev_hiddens[i]

                # save state for next time step
                prev_hiddens[i] = hidden

            # apply attention using the last layer's hidden state
            if self.attention is not None:
                assert attn_scores is not None
                #out, attn_scores[:, j, :] = self.attention(
                #    hidden, encoder_outs, encoder_padding_mask
                #)
                query = torch.unsqueeze(hidden, 0) # T(=1) x B x C
                attn_out = self.attention(
                    query=query,
                    key=encoder_outs,
                    value=encoder_outs,
                    key_padding_mask=encoder_padding_mask,
                    incremental_state=incremental_state,
                    static_kv=True,
                    need_weights=(not self.training and self.need_attn),
                    need_head_weights=False,
                )
                out = attn_out[0].squeeze(0) # B x C
                if (not self.training and self.need_attn):
                    attn_scores[:, j, :] = attn_out[1].squeeze(1).transpose(0, 1) # B x SRCLEN
            else:
                out = hidden
            if self.args.decoder_hidden_state_in_output:
                out = self.hidden_attn_combination_fc(torch.cat((out, hidden), dim=-1))
            out = self.dropout_out_module(out)

            # input feeding
            if input_feed is not None:
                input_feed = out

            # save final output
            outs.append(out)

        # Stack all the necessary tensors together and store
        prev_hiddens_tensor = torch.stack(prev_hiddens)
        cache_state = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {
                "prev_hiddens": prev_hiddens_tensor,
                "input_feed": input_feed,
            },
        )
        self.set_incremental_state(incremental_state, "cached_state", cache_state)

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        if hasattr(self, "additional_fc") and self.adaptive_softmax is None:
            x = self.additional_fc(x)
            x = self.dropout_out_module(x)
        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        if not self.training and self.need_attn and self.attention is not None:
            assert attn_scores is not None
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None
        return x, attn_scores

    def output_layer(self, x):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = self.fc_out(x)
        return x

    def get_cached_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
    ) -> Tuple[List[Tensor], List[Tensor], Optional[Tensor]]:
        cached_state = self.get_incremental_state(incremental_state, "cached_state")
        assert cached_state is not None
        prev_hiddens_ = cached_state["prev_hiddens"]
        assert prev_hiddens_ is not None
        prev_hiddens = [prev_hiddens_[i] for i in range(self.num_layers)]
        input_feed = cached_state[
            "input_feed"
        ]  # can be None for decoder-only language models
        return prev_hiddens, input_feed

    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        if incremental_state is None or len(incremental_state) == 0:
            return
        prev_hiddens, input_feed = self.get_cached_state(incremental_state)
        prev_hiddens = [p.index_select(0, new_order) for p in prev_hiddens]
        if input_feed is not None:
            input_feed = input_feed.index_select(0, new_order)
        cached_state_new = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {
                "prev_hiddens": torch.stack(prev_hiddens),
                "input_feed": input_feed,
            },
        )
        self.set_incremental_state(incremental_state, "cached_state", cached_state_new),
        return

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.max_target_positions

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

def GRUCell(input_size, hidden_size, **kwargs):
    m = nn.GRUCell(input_size, hidden_size, **kwargs)
#    for name, param in m.named_parameters():
#        if "weight" in name or "bias" in name:
#            param.data.uniform_(-0.1, 0.1)
    return m

DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)


@register_model("transformer_gru")
class TransformerGRUModel(FairseqEncoderDecoderModel):
    """
    Transformer encder GRU decoder 
    
    Args:
        encoder (TransformerEncoder): the encoder
        decoder (GRUDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
        self.supports_align_args = True

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                            help='decoder output dimension (extra linear layer '
                                 'if different from decoder embed dim')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        parser.add_argument('--checkpoint-activations', action='store_true',
                            help='checkpoint activations at each layer, which saves GPU '
                                 'memory usage at the cost of some additional compute')
        parser.add_argument('--offload-activations', action='store_true',
                            help='checkpoint activations at each layer, then save to gpu. Sets --checkpoint-activations.')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                            help='scalar quantization noise and scalar quantization at training time')
        # args for Fully Sharded Data Parallel (FSDP) training
        parser.add_argument(
            '--min-params-to-wrap', type=int, metavar='D', default=DEFAULT_MIN_PARAMS_TO_WRAP,
            help=(
                'minimum number of params for a layer to be wrapped with FSDP() when '
                'training with --ddp-backend=fully_sharded. Smaller values will '
                'improve memory efficiency, but may make torch.distributed '
                'communication less efficient due to smaller input sizes. This option '
                'is set to 0 (i.e., always wrap) when --checkpoint-activations or '
                '--offload-activations are passed.'
            )
        )
        
        parser.add_argument('--decoder-freeze-embed', action='store_true',
                            help='freeze decoder embeddings')
        parser.add_argument('--decoder-hidden-size', type=int, metavar='N',
                            help='decoder hidden size')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N',
                            help='decoder output embedding dimension')
        parser.add_argument('--decoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for decoder input embedding')
        parser.add_argument('--decoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for decoder output')
        parser.add_argument('--decoder-hidden-state-in-output', action='store_true',
                            help='include the hidden state in the computation of the output (Luong et al. 2015) ')

        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )
        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        if not args.share_all_embeddings:
            min_params_to_wrap = getattr(
                args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
            )
            # fsdp_wrap is a no-op when --ddp-backend != fully_sharded
            encoder = fsdp_wrap(encoder, min_num_params=min_params_to_wrap)
            decoder = fsdp_wrap(decoder, min_num_params=min_params_to_wrap)
        return cls(args, encoder, decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        #return TransformerDecoder(
        #    args,
        #    tgt_dict,
        #    embed_tokens,
        #    no_encoder_attn=getattr(args, "no_cross_attention", False),
        #)
        return GRUDecoder(
            args=args,
            dictionary=tgt_dict,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attention=not getattr(args, "no_cross_attention", False),
            encoder_output_units=getattr(args, "encoder_embed_dim", None),
            pretrained_embed=embed_tokens,
            share_input_output_embed=args.share_decoder_input_output_embed,
            adaptive_softmax_cutoff=(
                utils.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == "adaptive_loss"
                else None
            ),
            max_target_positions=args.max_target_positions,
            residuals=False,)

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            #features_only=features_only,
            #alignment_layer=alignment_layer,
            #alignment_heads=alignment_heads,
            #src_lengths=src_lengths,
            #return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)

@register_model_architecture("transformer_gru", "transformer_gru")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)

    args.decoder_freeze_embed = getattr(args, "decoder_freeze_embed", False)
    args.decoder_hidden_size = getattr( args, "decoder_hidden_size", args.decoder_embed_dim)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 512)
    args.decoder_dropout_in = getattr(args, "decoder_dropout_in", args.dropout)
    args.decoder_dropout_out = getattr(args, "decoder_dropout_out", args.dropout)
    args.decoder_hidden_state_in_output = getattr(args, "decoder_hidden_state_in_output", False)
    

@register_model_architecture("transformer_gru", "transformer_gru_iwslt_de_en")
def transformer_gru_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    base_architecture(args)