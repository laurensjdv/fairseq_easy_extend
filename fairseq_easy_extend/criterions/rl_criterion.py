import math

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from torch import Tensor
# from bert_score import BERTScorer


from dataclasses import dataclass, field

from sacrebleu.metrics import BLEU, CHRF
from comet import download_model, load_from_checkpoint
import sacremoses


@dataclass
class RLCriterionConfig(FairseqDataclass):
    sentence_level_metric: str = field(
        default="bleu", metadata={"help": "sentence level metric"}
    )
    temperature: float = field(
        default=1.0, metadata={"help": "temperature for sampling"}
    )


@register_criterion("rl_loss", dataclass=RLCriterionConfig)
class RLCriterion(FairseqCriterion):
    def __init__(self, task, sentence_level_metric, temperature):
        super().__init__(task)
        self.metric = sentence_level_metric
        self.tgt_dict = task.tgt_dict
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = temperature
        self.detokenizer = sacremoses.MosesDetokenizer(lang="en")
        self.bleu = BLEU(effective_order=True)
        self.chrf = CHRF()
        # self.bertscorer = BERTScorer(lang="en", rescale_with_baseline=True)
        # self.comet_model = load_from_checkpoint(
        #     download_model("Unbabel/wmt22-comet-da")
        # )

    def _compute_loss(
        self,
        src_tokens,
        outputs,
        targets,
        masks=None,
        label_smoothing=0.0,
        name="loss",
        factor=1.0,
    ):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len
        """
        bsz = outputs.size(0)
        seq_len = outputs.size(1)
        vocab_size = outputs.size(2)

        with torch.no_grad():
            probs = F.softmax(outputs, dim=-1).view(-1, vocab_size) / self.temperature
            sample_idx = torch.multinomial(probs, 1, replacement=True).view(
                bsz, seq_len
            )
            # sampled_sentence_string = [
            #     self.tgt_dict.string(sample) for sample in sample_idx
            # ]
            # target_sentence_string = [
            #     self.tgt_dict.string(targets) for sample in targets
            # ]
            sampled_sentence_string = [
                self.detokenizer.detokenize(
                    self.tgt_dict.string(sample).split(), return_str=True
                )
                for sample in sample_idx
            ]
            target_sentence_string = [
                self.detokenizer.detokenize(
                    self.tgt_dict.string(sample).split(), return_str=True
                )
                for sample in targets
            ]

        # print(len(sampled_sentence_string))
        # print(len(target_sentence_string))
        with torch.no_grad():
            if self.metric == "constant":
                R = 1
            elif self.metric == "bleu":
                
                # R = bleu.sentence_score(
                #     [sampled_sentence_string], [[target_sentence_string]]
                # ).score
                R = torch.tensor(
                    [
                        [self.bleu.sentence_score(sample, [target]).score] * seq_len
                        for sample, target in zip(
                            sampled_sentence_string, target_sentence_string
                        )
                    ]
                )

            elif self.metric == "chrf":
                # R = chrf.corpus_score(
                #     [sampled_sentence_string], [[target_sentence_string]]
                # ).score
                R = torch.tensor(
                    [
                        [self.chrf.sentence_score(sample, [target]).score] * seq_len
                        for sample, target in zip(
                            sampled_sentence_string, target_sentence_string
                        )
                    ]
                )
            # elif self.metric == 'bert':
            #     _, _, F1 = self.bertscorer.score(sampled_sentence_string, target_sentence_string)
            #     # print(F1.size())
            #     R = torch.tensor([[F1s] * seq_len for F1s in F1])
            # reward = torch.tensor([[R] * seq_len] * bsz).to(self.device)
            reward = R.to(self.device)

        # print(reward.size())

        # padding mask, do not remove
        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]
            reward, sample_idx = reward[masks], sample_idx[masks]

        log_probs = F.log_softmax(outputs, dim=-1)
        log_probs_of_samples = torch.gather(log_probs, 1, sample_idx.unsqueeze(1))
        loss = -log_probs_of_samples * reward
        loss = loss.mean()
        nll_loss = loss

        loss = loss * factor

        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}

    def _custom_loss(self, loss, name="loss", factor=1.0):
        return {"name": name, "loss": loss, "factor": factor}

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]

        outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens)
        # get loss only on tokens, not on lengths
        outs = outputs["word_ins"].get("out", None)
        masks = outputs["word_ins"].get("mask", None)

        loss_dict = self._compute_loss(src_tokens, outs, tgt_tokens, masks)
        loss = loss_dict["loss"]
        nll_loss = loss_dict["nll_loss"]

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.detach(),
            "nll_loss": nll_loss.detach(),
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nll_loss = utils.item(sum(log.get("nll_loss", 0) for log in logging_outputs))

        metrics.log_scalar(
            "loss", loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size / math.log(2) if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
