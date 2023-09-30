"""References:
[1] https://colab.research.google.com/drive/1lobspU9b7dTO_HuoX-3nibZspTwfa5aX?usp=sharing#scrollTo=OC2Hwa712_2x
[2] Vinyals, O., Fortunato, M., & Jaitly, N. (2015). Pointer networks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Seq2Seq(nn.Module):
    def __init__(
            self, encoder, encoder_key, decoder, decoder_key, tokenizer, config,
            beam_size=None, max_source_length=None, max_target_length=None,
            sos_id=None, eos_id=None,
        ):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.encoder_key = encoder_key
        self.decoder = decoder
        self.decoder_key = decoder_key
        self.config = config

        if self.decoder_key == 'transformer':
            self.register_buffer(
                "bias", torch.tril(torch.ones((1024, 1024), dtype=torch.uint8))
            )
        else:
            self.register_buffer(
                "bias", torch.tril(torch.ones((1024, 1024), dtype=torch.uint8)).view(1, 1024, 1024)
            )

        if self.encoder_key == self.decoder_key == 'graphcodebert':
            self.register_buffer(
                "bias", torch.tril(torch.ones((512, 512), dtype=torch.uint8)).view(1, 512, 512)
            )

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.encoder.embeddings.word_embeddings.weight

        self.lsm = nn.LogSoftmax(dim=-1)
        self.ignore_cross_entropy_index = 1
    
        self.beam_size = beam_size
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.sos_id = sos_id
        self.eos_id = eos_id

    def forward(self, source_ids, target_ids=None):
        if target_ids is None:
            return self.generate(source_ids)

        if self.decoder_key == 'transformer':
            source_mask = source_ids.ne(1)
            encoder_output = self.encoder(source_ids, attention_mask=source_mask)
            encoder_output = encoder_output[0].permute([1, 0, 2]).contiguous()
            attn_mask = -1e4 * (1 - self.bias[:target_ids.shape[1], :target_ids.shape[1]])
            tgt_embeddings = self.encoder.embeddings(target_ids).permute([1, 0, 2]).contiguous()
            out = self.decoder(tgt_embeddings, encoder_output, tgt_mask=attn_mask,
                               memory_key_padding_mask=torch.logical_not(source_mask.bool()))
            hidden_states = torch.tanh(self.dense(out)).permute([1, 0, 2]).contiguous()
            lm_logits = self.lm_head(hidden_states)

        if self.decoder_key == self.encoder_key and self.decoder_key in ['unixcoder', 'graphcodebert']:
            source_mask = source_ids.ne(1)[:, None, :] * source_ids.ne(1)[:, :, None]
            encoder_output = self.encoder(source_ids, attention_mask=source_mask, use_cache=True)
            ids = torch.cat((source_ids, target_ids), -1)
            target_mask = self.bias[:, source_ids.size(-1) :ids.size(-1), :ids.size(-1)].bool()
            target_mask = target_mask & ids[:, None, :].ne(1)
            decoder_output = self.decoder(target_ids, attention_mask=target_mask,
                                past_key_values=encoder_output.past_key_values).last_hidden_state
            lm_logits = self.lm_head(decoder_output)

        # Shift so that tokens < n predict n
        active_loss = target_ids[..., 1:].ne(1).view(-1)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_cross_entropy_index)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                        shift_labels.view(-1)[active_loss])
        outputs = loss, loss * active_loss.sum(), active_loss.sum()
        return outputs
    
    def generate(self, source_ids):
        if self.decoder_key == 'transformer':
            source_mask = source_ids.ne(1)
            encoder_output = self.encoder(source_ids, attention_mask=source_mask, use_cache=True)
            encoder_output = encoder_output[0].permute([1, 0, 2]).contiguous()
            preds = []
            pad = torch.cuda.LongTensor(1).fill_(-999)

            for i in range(source_ids.shape[0]):
                context = encoder_output[:, i:i + 1]
                context_mask = source_mask[i:i + 1, :]
                beam = Beam(self.beam_size, self.sos_id, self.eos_id)
                input_ids = beam.getCurrentState()
                context = context.repeat(1, self.beam_size, 1)
                context_mask = context_mask.repeat(self.beam_size, 1)
                for _ in range(self.max_target_length):
                    if beam.done():
                        break
                    attn_mask = -1e4 * (1 - self.bias[:input_ids.shape[1], :input_ids.shape[1]])
                    tgt_embeddings = self.encoder.embeddings(input_ids).permute([1, 0, 2]).contiguous()
                    out = self.decoder(tgt_embeddings, context, tgt_mask=attn_mask,
                                       memory_key_padding_mask=torch.logical_not(context_mask.bool()))
                    out = torch.tanh(self.dense(out))
                    hidden_states = out.permute([1, 0, 2]).contiguous()[:, -1, :]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                    input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:self.beam_size]
                pred = [torch.cat([x.view(-1) for x in p] + \
                                  [pad] * (self.max_target_length - len(p))).view(1, -1) for p in pred]
                preds.append(torch.cat(pred, 0).unsqueeze(0))

        if self.decoder_key == self.encoder_key and self.decoder_key in ['unixcoder', 'graphcodebert']:
            source_mask = source_ids.ne(1)[:, None, :] * source_ids.ne(1)[:, :, None]
            encoder_output = self.encoder(source_ids, attention_mask=source_mask, use_cache=True)
            preds = []
            zero = torch.cuda.LongTensor(1).fill_(0)
            source_len = list(source_ids.ne(1).sum(-1).cpu().numpy())
            for i in range(source_ids.shape[0]):
                context = [[x[i: i+1, :, :source_len[i]].repeat(self.beam_size, 1, 1, 1) for x in y] \
                            for y in encoder_output.past_key_values]
                beam = Beam(self.beam_size, self.sos_id, self.eos_id)
                input_ids = beam.getCurrentState()
                context_ids = source_ids[i:i+1, :source_len[i]].repeat(self.beam_size, 1)
                for _ in range(self.max_target_length):
                    if beam.done():
                        break
                    ids = torch.cat((context_ids, input_ids), -1)
                    target_mask = self.bias[:, context_ids.size(-1):ids.size(-1), :ids.size(-1)].bool()
                    target_mask = target_mask & ids[:, None, :].ne(1)
                    out = self.decoder(input_ids, attention_mask=target_mask,
                                        past_key_values=context).last_hidden_state
                    hidden_states = out[:, -1, :]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                    input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:self.beam_size]
                pred = [torch.cat([x.view(-1) for x in p] + \
                                  [zero] * (self.max_target_length - len(p))).view(1, -1) for p in pred]
                preds.append(torch.cat(pred, 0).unsqueeze(0))

        preds = torch.cat(preds, 0)
        return preds


class PointerGeneratedSeq2Seq(nn.Module):
    def __init__(
            self, encoder, encoder_key, decoder, tokenizer, config, beam_size=None,
            max_source_length=None, max_target_length=None, sos_id=None, eos_id=None,
        ):
        super(PointerGeneratedSeq2Seq, self).__init__()
        self.encoder = encoder
        self.encoder_key = encoder_key
        self.decoder = decoder
        self.decoder_key = 'transformer'
        self.config = config

        self.register_buffer(
            "bias", torch.tril(torch.ones((1024, 1024), dtype=torch.uint8))
        )

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, max_source_length, bias=False)
        lm_token_ids = torch.tensor(
            tokenizer.convert_tokens_to_ids([f"<unk-{i}>" for i in range(max_source_length)]),
            dtype=torch.long,
        )
        self.lm_head.weight = nn.Parameter(
            self.encoder.embeddings.word_embeddings.weight.data[lm_token_ids], 
            requires_grad=True
        )
        self.lsm = nn.LogSoftmax(dim=-1)
        self.ignore_cross_entropy_index = max_source_length
    
        self.beam_size = beam_size
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.sos_id = sos_id
        self.eos_id = eos_id

    def forward(self, source_ids, target_ids=None):
        if target_ids is None:
            return self.generate(source_ids)

        source_mask = source_ids.ne(1)
        encoder_output = self.encoder(source_ids, attention_mask=source_mask)
        encoder_output = encoder_output[0].permute([1, 0, 2]).contiguous()
        attn_mask = -1e4 * (1 - self.bias[:target_ids.shape[1], :target_ids.shape[1]])
        tgt_embeddings = self.encoder.embeddings(target_ids).permute([1, 0, 2]).contiguous()
        out = self.decoder(tgt_embeddings, encoder_output, tgt_mask=attn_mask,
                            memory_key_padding_mask=torch.logical_not(source_mask.bool()))
        hidden_states = torch.tanh(self.dense(out)).permute([1, 0, 2]).contiguous()
        lm_logits = self.lm_head(hidden_states)

        # Shift so that tokens < n predict n
        active_loss = target_ids[..., 1:].ne(1).view(-1)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_cross_entropy_index)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                        shift_labels.view(-1)[active_loss])
        outputs = loss, loss * active_loss.sum(), active_loss.sum()
        return outputs
    
    def generate(self, source_ids):
        source_mask = source_ids.ne(1)
        encoder_output = self.encoder(source_ids, attention_mask=source_mask, use_cache=True)
        encoder_output = encoder_output[0].permute([1, 0, 2]).contiguous()
        preds = []
        pad = torch.cuda.LongTensor(1).fill_(-999)

        for i in range(source_ids.shape[0]):
            context = encoder_output[:, i:i + 1]
            context_mask = source_mask[i:i + 1, :]
            beam = Beam(self.beam_size, self.sos_id, self.eos_id)
            input_ids = beam.getCurrentState()
            context = context.repeat(1, self.beam_size, 1)
            context_mask = context_mask.repeat(self.beam_size, 1)
            for _ in range(self.max_target_length):
                if beam.done():
                    break
                attn_mask = -1e4 * (1 - self.bias[:input_ids.shape[1], :input_ids.shape[1]])
                tgt_embeddings = self.encoder.embeddings(input_ids).permute([1, 0, 2]).contiguous()
                out = self.decoder(tgt_embeddings, context, tgt_mask=attn_mask,
                                    memory_key_padding_mask=torch.logical_not(context_mask.bool()))
                out = torch.tanh(self.dense(out))
                hidden_states = out.permute([1, 0, 2]).contiguous()[:, -1, :]
                out = self.lsm(self.lm_head(hidden_states)).data
                beam.advance(out)
                input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
            hyp = beam.getHyp(beam.getFinal())
            pred = beam.buildTargetTokens(hyp)[:self.beam_size]
            pred = [torch.cat([x.view(-1) for x in p] + \
                                [pad] * (self.max_target_length - len(p))).view(1, -1) for p in pred]
            preds.append(torch.cat(pred, 0).unsqueeze(0))

        preds = torch.cat(preds, 0)
        return preds


class Beam(object):
	def __init__(self, size, sos, eos):
		self.size = size
		self.tt = torch.cuda
		# The score for each translation on the beam.
		self.scores = self.tt.FloatTensor(size).zero_()
		# The backpointers at each time-step.
		self.prevKs = []
		# The outputs at each time-step.
		self.nextYs = [self.tt.LongTensor(size).fill_(0)]
		self.nextYs[0][0] = sos
		# Has EOS topped the beam yet.
		self._eos = eos
		self.eosTop = False
		# Time and k pair for finished.
		self.finished = []

	def getCurrentState(self):
		"Get the outputs for the current timestep."
		batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
		return batch

	def getCurrentOrigin(self):
		"Get the backpointers for the current timestep."
		return self.prevKs[-1]

	def advance(self, wordLk):
		numWords = wordLk.size(1)

		# Sum the previous scores.
		if len(self.prevKs) > 0:
			beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

			# Don't let EOS have children.
			for i in range(self.nextYs[-1].size(0)):
				if self.nextYs[-1][i] == self._eos:
					beamLk[i] = -1e20
		else:
			beamLk = wordLk[0]
		flatBeamLk = beamLk.view(-1)
		bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

		self.scores = bestScores

		# bestScoresId is flattened beam x word array, so calculate which
		# word and beam each score came from
		prevK = bestScoresId // numWords
		self.prevKs.append(prevK)
		self.nextYs.append((bestScoresId - prevK * numWords))

		for i in range(self.nextYs[-1].size(0)):
			if self.nextYs[-1][i] == self._eos:
				s = self.scores[i]
				self.finished.append((s, len(self.nextYs) - 1, i))

		# End condition is when top-of-beam is EOS and no global score.
		if self.nextYs[-1][0] == self._eos:
			self.eosTop = True

	def done(self):
		return self.eosTop and len(self.finished) >=self.size

	def getFinal(self):
		if len(self.finished) == 0:
			self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
		self.finished.sort(key=lambda a: -a[0])
		if len(self.finished) != self.size:
			unfinished=[]
			for i in range(self.nextYs[-1].size(0)):
				if self.nextYs[-1][i] != self._eos:
					s = self.scores[i]
					unfinished.append((s, len(self.nextYs) - 1, i)) 
			unfinished.sort(key=lambda a: -a[0])
			self.finished+=unfinished[:self.size-len(self.finished)]
		return self.finished[:self.size]

	def getHyp(self, beam_res):
		hyps=[]
		for _,timestep, k in beam_res:
			hyp = []
			for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
				hyp.append(self.nextYs[j+1][k])
				k = self.prevKs[j][k]
			hyps.append(hyp[::-1])
		return hyps
	
	def buildTargetTokens(self, preds):
		sentence=[]
		for pred in preds:
			tokens = []
			for tok in pred:
				if tok==self._eos:
					break
				tokens.append(tok)
			sentence.append(tokens)
		return sentence
