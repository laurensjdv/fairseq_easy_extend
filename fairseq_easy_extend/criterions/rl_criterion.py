import math

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from torch import Tensor

from dataclasses import dataclass, field

from sacrebleu.metrics import BLEU, CHRF
from comet import download_model, load_from_checkpoint


@dataclass
class RLCriterionConfig(FairseqDataclass):
    sentence_level_metric: str = field(
        default="bleu", metadata={"help": "sentence level metric"}
    )


@register_criterion("rl_loss", dataclass=RLCriterionConfig)
class RLCriterion(FairseqCriterion):
    def __init__(self, task, sentence_level_metric):
        super().__init__(task)
        self.metric = sentence_level_metric
        self.tgt_dict = task.tgt_dict
        self.comet_model = load_from_checkpoint(
            download_model("Unbabel/wmt22-comet-da")
        )

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

        # padding mask, do not remove
        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        # we take a softmax over outputs
        # argmax over the softmax \ sampling (e.g. multinomial)
        # sampled_sentence = [4, 17, 18, 19, 20]
        # sampled_sentence_string = tgt_dict.string([4, 17, 18, 19, 20])
        # see dictionary class of fairseq
        # target_sentence = "I am a sentence"
        # with torch.no_grad()
        # R(*) = eval_metric(sampled_sentence_string, target_sentence)
        # R(*) is a number, BLEU, сhrf, etc.

        # loss = -log_prob(outputs)*R()
        # loss = loss.mean()

        prob = F.softmax(outputs, dim=-1)
        log_prob = torch.log(prob)

        # multinomial distribution
        # dist = torch.multinomial(log_prob.exp(),1)
        # sampled_sentence = dist.sample()
        sampled_sentence = torch.multinomial(prob, 1).squeeze(-1)
        sampled_sentence_string = self.tgt_dict.string(sampled_sentence)

        target_sentence = self.tgt_dict.string(targets)
        with torch.no_grad():
            if self.metric == "bleu":
                bleu = BLEU()
                R = bleu.corpus_score(
                    [sampled_sentence_string], [[target_sentence]]
                ).score
            elif self.metric == "chrf":
                chrf = CHRF()
                R = chrf.corpus_score(
                    [sampled_sentence_string], [[target_sentence]]
                ).score
            elif self.metric == "comet":
                data = {"src": src_tokens, "mt": sampled_sentence, "ref": targets}
                R = self.comet_model.predict(data)

        loss = -log_prob * R
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
        outputs = outputs["word_ins"]
        masks = (outputs.get("mask", None),)
        loss = self._compute_loss(src_tokens, outputs, tgt_tokens, masks)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.detach(),
            "nll_loss": loss.detach(),
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
