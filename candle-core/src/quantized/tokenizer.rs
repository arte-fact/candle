use crate::quantized::gguf_file;
use crate::{Context, Error, Result};
use std::collections::HashSet;
use tokenizers::{
    decoders::{
        byte_fallback::ByteFallback, byte_level::ByteLevel as ByteLevelDecoder, fuse::Fuse,
        sequence::Sequence as DecoderSequence, strip::Strip, DecoderWrapper,
    },
    models::bpe::{Vocab, BPE},
    normalizers::{replace::Replace, unicode::NFC, NormalizerWrapper},
    pre_tokenizers::{
        byte_level::ByteLevel as ByteLevelPre,
        sequence::Sequence,
        split::{Split, SplitPattern},
        PreTokenizerWrapper,
    },
    processors::sequence::Sequence as ProcessorSequence,
    processors::{byte_level::ByteLevel as ByteLevelProcessor, PostProcessorWrapper},
    tokenizer::SplitDelimiterBehavior,
    AddedToken, Tokenizer,
};

pub trait TokenizerFromGguf: Sized {
    fn from_gguf(ct: &gguf_file::Content) -> Result<Self>;
}

fn metadata_value<'a>(ct: &'a gguf_file::Content, key: &str) -> Result<&'a gguf_file::Value> {
    ct.metadata
        .get(key)
        .with_context(|| format!("missing GGUF metadata key `{key}`"))
}

fn gguf_value_to_u32(v: &gguf_file::Value) -> Result<u32> {
    use gguf_file::Value::*;
    match v {
        U8(v) => Ok(*v as u32),
        I8(v) => Ok(*v as u32),
        U16(v) => Ok(*v as u32),
        I16(v) => Ok(*v as u32),
        U32(v) => Ok(*v),
        I32(v) => Ok(*v as u32),
        U64(v) => Ok(*v as u32),
        I64(v) => Ok(*v as u32),
        _ => crate::bail!("expected numeric value for token type/id, got {v:?}"),
    }
}

fn value_to_string_array(v: &gguf_file::Value, name: &str) -> Result<Vec<String>> {
    let arr = v
        .to_vec()
        .with_context(|| format!("`{name}` is not an array"))?;
    arr.iter()
        .map(|v| {
            v.to_string()
                .map(|s| s.to_string())
                .with_context(|| format!("`{name}` element is not a string: {v:?}"))
        })
        .collect()
}

fn merges_from_value(v: &gguf_file::Value) -> Result<Vec<(String, String)>> {
    value_to_string_array(v, "tokenizer.ggml.merges")?
        .into_iter()
        .map(|m| {
            m.split_once(' ')
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .ok_or_else(|| Error::msg(format!("invalid merge entry `{m}`")))
        })
        .collect()
}

struct Pipeline {
    normalizer: Option<NormalizerWrapper>,
    pretokenizer: Option<PreTokenizerWrapper>,
    decoder: Option<DecoderWrapper>,
    post_processor: Option<PostProcessorWrapper>,
}

impl Pipeline {
    fn apply(self, tokenizer: &mut Tokenizer) {
        if let Some(norm) = self.normalizer {
            tokenizer.with_normalizer(Some(norm));
        }
        if let Some(pt) = self.pretokenizer {
            tokenizer.with_pre_tokenizer(Some(pt));
        }
        if let Some(dec) = self.decoder {
            tokenizer.with_decoder(Some(dec));
        }
        if let Some(pp) = self.post_processor {
            tokenizer.with_post_processor(Some(pp));
        }
    }
}

fn pre_tokenizer_sequence(regex: &str, byte_level: ByteLevelPre) -> Result<PreTokenizerWrapper> {
    let split = Split::new(
        SplitPattern::Regex(regex.to_string()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(Error::wrap)?;
    Ok(Sequence::new(vec![split.into(), byte_level.into()]).into())
}

/// Build the SentencePiece-style pipeline used by Gemma / LLaMA-1 family models.
///
/// Mirrors the HuggingFace `gemma` tokenizer.json:
///   - normalizer: Replace " " → "▁"
///   - pre-tokenizer: none
///   - decoder: Sequence([Replace("▁"," "), ByteFallback, Fuse, Strip(" ", 1, 0)])
fn gemma_pipeline() -> Result<Pipeline> {
    let normalizer = Replace::new(" ", "\u{2581}").map_err(Error::wrap)?;
    let dec_replace = Replace::new("\u{2581}", " ").map_err(Error::wrap)?;
    let decoder = DecoderSequence::new(vec![
        DecoderWrapper::Replace(dec_replace),
        DecoderWrapper::ByteFallback(ByteFallback::new()),
        DecoderWrapper::Fuse(Fuse::new()),
        DecoderWrapper::Strip(Strip::new(' ', 1, 0)),
    ]);
    Ok(Pipeline {
        normalizer: Some(NormalizerWrapper::Replace(normalizer)),
        pretokenizer: None,
        decoder: Some(DecoderWrapper::Sequence(decoder)),
        post_processor: None,
    })
}

fn pipeline_from_pre(pre: &str) -> Result<Pipeline> {
    const REGEX_QWEN2: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const REGEX_LLAMA3: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    Ok(match pre {
        // Matches Qwen2 tokenizer.json settings
        "qwen2" => Pipeline {
            normalizer: Some(NFC.into()),
            pretokenizer: Some(pre_tokenizer_sequence(
                REGEX_QWEN2,
                ByteLevelPre::new(false, false, false),
            )?),
            decoder: Some(ByteLevelDecoder::new(false, false, false).into()),
            post_processor: Some(ByteLevelProcessor::new(false, false, false).into()),
        },
        // Matches Smaug/Llama3 style byte-level BPE
        "smaug-bpe" | "lfm2" | "llama3" => Pipeline {
            normalizer: None,
            pretokenizer: Some(pre_tokenizer_sequence(
                REGEX_LLAMA3,
                ByteLevelPre::new(false, true, false),
            )?),
            decoder: Some(ByteLevelDecoder::new(true, true, true).into()),
            post_processor: Some(ByteLevelProcessor::new(true, false, true).into()),
        },
        // Default GPT-2 style BPE
        _ => Pipeline {
            normalizer: None,
            pretokenizer: Some(ByteLevelPre::default().into()),
            decoder: Some(ByteLevelDecoder::default().into()),
            post_processor: Some(ByteLevelProcessor::default().into()),
        },
    })
}

fn template_processor(
    tokens: &[String],
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    add_bos: bool,
    add_eos: bool,
) -> Option<PostProcessorWrapper> {
    if (!add_bos && !add_eos) || tokens.is_empty() {
        return None;
    }

    let bos = bos_id.and_then(|id| tokens.get(id as usize)).cloned();
    let eos = eos_id.and_then(|id| tokens.get(id as usize)).cloned();

    let mut specials = Vec::new();
    if add_bos {
        let bos_id = bos_id?;
        let bos_tok = bos.clone()?;
        specials.push((bos_tok.clone(), bos_id));
    }
    if add_eos {
        let eos_id = eos_id?;
        let eos_tok = eos.clone()?;
        specials.push((eos_tok.clone(), eos_id));
    }

    let mut single = Vec::new();
    if add_bos {
        single.push(bos.clone()?);
    }
    single.push("$0".to_string());
    if add_eos {
        single.push(eos.clone()?);
    }

    let mut pair = Vec::new();
    if add_bos {
        pair.push(format!("{}:0", bos.clone()?));
    }
    pair.push("$A:0".to_string());
    if add_eos {
        pair.push(format!("{}:0", eos.clone()?));
    }
    if add_bos {
        pair.push(format!("{}:1", bos.clone()?));
    }
    pair.push("$B:1".to_string());
    if add_eos {
        pair.push(format!("{}:1", eos.clone()?));
    }

    let proc = tokenizers::processors::template::TemplateProcessing::builder()
        .try_single(single)
        .ok()?
        .try_pair(pair)
        .ok()?
        .special_tokens(specials)
        .build()
        .ok()?;

    Some(PostProcessorWrapper::Template(proc))
}

impl TokenizerFromGguf for Tokenizer {
    fn from_gguf(ct: &gguf_file::Content) -> Result<Self> {
        let model_kind = metadata_value(ct, "tokenizer.ggml.model")?
            .to_string()?
            .to_lowercase();
        let is_spm = matches!(
            model_kind.as_str(),
            "llama" | "gemma" | "gemma2" | "gemma3" | "gemma4"
        );
        if model_kind != "gpt2" && !is_spm {
            crate::bail!("unsupported tokenizer model `{model_kind}`");
        }

        let tokens = value_to_string_array(
            metadata_value(ct, "tokenizer.ggml.tokens")?,
            "tokenizer.ggml.tokens",
        )?;
        let vocab: Vocab = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        let merges = merges_from_value(metadata_value(ct, "tokenizer.ggml.merges")?)?;

        let mut builder = BPE::builder().vocab_and_merges(vocab, merges);

        // Some converters write `unk_token_id`, others (gemma, gemma4) write `unknown_token_id`.
        let unk_meta = metadata_value(ct, "tokenizer.ggml.unk_token_id")
            .or_else(|_| metadata_value(ct, "tokenizer.ggml.unknown_token_id"));
        if let Ok(val) = unk_meta {
            let token_id = gguf_value_to_u32(val)?;
            if let Some(token) = tokens.get(token_id as usize) {
                builder = builder.unk_token(token.clone());
            }
        }

        if let Ok(val) = metadata_value(ct, "tokenizer.ggml.byte_fallback") {
            builder = builder.byte_fallback(val.to_bool()?);
        } else if is_spm {
            // SentencePiece tokenizers always use byte fallback even when the
            // metadata flag is absent (HF gemma tokenizer.json sets it).
            builder = builder.byte_fallback(true);
        }

        if let Ok(val) = metadata_value(ct, "tokenizer.ggml.ignore_merges") {
            builder = builder.ignore_merges(val.to_bool()?);
        }

        let bpe = builder.build().map_err(Error::wrap)?;
        let mut tokenizer = Tokenizer::new(bpe);

        let pre = metadata_value(ct, "tokenizer.ggml.pre")
            .and_then(|v| v.to_string())
            .map(|s| s.to_string())
            .unwrap_or_else(|_| "gpt2".to_string());
        let pipeline = if is_spm {
            // SentencePiece-style models (gemma, llama) ignore tokenizer.ggml.pre
            // and use the Replace(" "→"▁") + ByteFallback decoder pipeline.
            gemma_pipeline()?
        } else {
            pipeline_from_pre(pre.as_str())?
        };
        let post_processor_base = pipeline.post_processor.clone();

        let add_bos = metadata_value(ct, "tokenizer.ggml.add_bos_token")
            .and_then(|v| v.to_bool())
            .unwrap_or(false);
        let add_eos = metadata_value(ct, "tokenizer.ggml.add_eos_token")
            .and_then(|v| v.to_bool())
            .unwrap_or(false);
        let bos_id = metadata_value(ct, "tokenizer.ggml.bos_token_id")
            .and_then(gguf_value_to_u32)
            .ok();
        let eos_id = metadata_value(ct, "tokenizer.ggml.eos_token_id")
            .and_then(gguf_value_to_u32)
            .ok();

        pipeline.apply(&mut tokenizer);

        // Compose existing post-processor with a template-based one if needed
        let template_pp = template_processor(&tokens, bos_id, eos_id, add_bos, add_eos);
        if template_pp.is_some() || post_processor_base.is_some() {
            let mut steps = Vec::new();
            if let Some(pp) = post_processor_base {
                steps.push(pp);
            }
            if let Some(tp) = template_pp {
                steps.push(tp);
            }
            let pp = if steps.len() == 1 {
                steps.pop().unwrap()
            } else {
                ProcessorSequence::new(steps).into()
            };
            tokenizer.with_post_processor(Some(pp));
        }

        // Mark special tokens so decode(skip_special_tokens = true) behaves as expected
        if let Ok(gguf_file::Value::Array(arr)) = metadata_value(ct, "tokenizer.ggml.token_type") {
            let mut specials = Vec::new();
            for (idx, v) in arr.iter().enumerate() {
                let ty = gguf_value_to_u32(v)?;
                // Aligns with llama_token_type: treat non-normal/non-byte tokens as special.
                let is_special = matches!(ty, 2..=5);
                if is_special {
                    if let Some(tok) = tokens.get(idx) {
                        specials.push(AddedToken::from(tok.clone(), true));
                    }
                }
            }
            if !specials.is_empty() {
                tokenizer.add_special_tokens(&specials);
            }
        }

        let mut explicit_specials = HashSet::new();
        for key in [
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.pad_token_id",
            "tokenizer.ggml.padding_token_id",
            "tokenizer.ggml.sep_token_id",
            "tokenizer.ggml.unk_token_id",
            "tokenizer.ggml.unknown_token_id",
            "tokenizer.ggml.mask_token_id",
        ] {
            if let Ok(val) = metadata_value(ct, key) {
                explicit_specials.insert(gguf_value_to_u32(val)?);
            }
        }
        if !explicit_specials.is_empty() {
            let specials: Vec<_> = explicit_specials
                .into_iter()
                .filter_map(|id| tokens.get(id as usize))
                .map(|tok| AddedToken::from(tok.clone(), true))
                .collect();
            if !specials.is_empty() {
                tokenizer.add_special_tokens(&specials);
            }
        }

        Ok(tokenizer)
    }
}
