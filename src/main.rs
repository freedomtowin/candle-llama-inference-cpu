// Conditional compilation for external crates based on features
#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

// Importing the helper module
pub mod helper;

// Importing necessary crates and modules
use anyhow::{bail, Error as E, Result};
use clap::{ValueEnum};

use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::utils::{apply_repeat_penalty};
use candle_transformers::models::llama as model;

use hf_hub::{api::sync::Api, Repo, RepoType};
use helper::{device, hub_load_safetensors};
use helper::{TokenOutputStream};
use std::io::Write;

use model::{Llama, LlamaConfig};

// Enum to specify model versions
#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum Which {
    V1,
    V2,
    V3,
    V3Instruct,
    #[value(name = "solar-10.7b")]
    Solar10_7B,
    #[value(name = "tiny-llama-1.1b-chat")]
    TinyLlama1_1BChat,
}

// Hardcoded configuration values
const EOS_TOKEN: &str = "";
const DEFAULT_PROMPT: &str = "My favorite theorem is ";
const CPU: bool = false;
const TEMPERATURE: f64 = 0.1;
const TOP_P: Option<f64> = None;
const TOP_K: Option<usize> = None;
const SEED: u64 = 299792458;
const SAMPLE_LEN: usize = 10000;
const NO_KV_CACHE: bool = false;
const PROMPT: Option<&str> = None;
const DTYPE: Option<&str> = None;
const TRACING: bool = false;
const MODEL_ID: Option<&str> = None;
const REVISION: Option<&str> = None;
const WHICH: Which = Which::V3Instruct;
const USE_FLASH_ATTN: bool = false;
const REPEAT_PENALTY: f32 = 1.1;
const REPEAT_LAST_N: usize = 128;

// Main function to run the application
fn main() -> Result<()> {
    use tokenizers::Tokenizer;

    // Select the computation device (CPU/GPU)
    let device = device(CPU)?;

    // Determine the data type (dtype) for computations
    let dtype = match DTYPE {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        Some(dtype) => bail!("Unsupported dtype {dtype}"),
        None => DType::F16,
    };

    // Load the model, tokenizer, cache, and configuration
    let (llama, tokenizer_filename, mut cache, config) = {
        let api = Api::new()?;
        let model_id = MODEL_ID.unwrap_or_else(|| match WHICH {
            Which::V1 => "Narsil/amall-7b",
            Which::V2 => "meta-llama/Llama-2-7b-hf",
            Which::V3 => "meta-llama/Meta-Llama-3-8B",
            Which::V3Instruct => "meta-llama/Meta-Llama-3-8B-Instruct",
            Which::Solar10_7B => "upstage/SOLAR-10.7B-v1.0",
            Which::TinyLlama1_1BChat => "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        }).to_string();

        println!("loading the model weights from {model_id}");
        let revision = REVISION.unwrap_or("main").to_string();
        let api = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

        // Retrieve tokenizer and config files from the repository
        let tokenizer_filename = api.get("tokenizer.json")?;
        let config_filename = api.get("config.json")?;
        let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
        let config = config.into_config(USE_FLASH_ATTN);

        // Load model weights based on the specified version
        let filenames = match WHICH {
            Which::V1 | Which::V2 | Which::V3 | Which::V3Instruct | Which::Solar10_7B => {
                hub_load_safetensors(api, "model.safetensors.index.json")?
            }
            Which::TinyLlama1_1BChat => vec![api.get("model.safetensors")?],
        };

        // Create a cache for the model
        let cache = model::Cache::new(!NO_KV_CACHE, dtype, &config, &device)?;

        // Load the model from the safetensors files
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
        (Llama::load(vb, &config)?, tokenizer_filename, cache, config)
    };

    // Initialize the tokenizer
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    // Retrieve the end-of-sequence (EOS) token ID
    let eos_token_id = config
        .eos_token_id
        .or_else(|| tokenizer.token_to_id(EOS_TOKEN));

    // Use the default prompt if none is provided
    let prompt = PROMPT.unwrap_or(DEFAULT_PROMPT);

    // Encode the prompt to token IDs
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    // Initialize the TokenOutputStream for streaming token output
    let mut tokenizer = TokenOutputStream::new(tokenizer);

    println!("starting the inference loop");
    print!("{prompt}");

    // Configure the logits processor for sampling
    let mut logits_processor = {
        let temperature = TEMPERATURE;
        let sampling = if temperature <= 0. {
            Sampling::ArgMax
        } else {
            match (TOP_K, TOP_P) {
                (None, None) => Sampling::All { temperature },
                (Some(k), None) => Sampling::TopK { k, temperature },
                (None, Some(p)) => Sampling::TopP { p, temperature },
                (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
            }
        };
        LogitsProcessor::from_sampling(SEED, sampling)
    };

    let mut start_gen = std::time::Instant::now();
    let mut index_pos = 0;
    let mut token_generated = 0;

    // Main inference loop for token generation
    for index in 0..SAMPLE_LEN {
        let (context_size, context_index) = if cache.use_kv_cache && index > 0 {
            (1, index_pos)
        } else {
            (tokens.len(), 0)
        };
        if index == 1 {
            start_gen = std::time::Instant::now()
        }
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
        let logits = llama.forward(&input, context_index, &mut cache)?;
        let logits = logits.squeeze(0)?;
        let logits = if REPEAT_PENALTY == 1. {
            logits
        } else {
            let start_at = tokens.len().saturating_sub(REPEAT_LAST_N);
            apply_repeat_penalty(
                &logits,
                REPEAT_PENALTY,
                &tokens[start_at..],
            )?
        };
        index_pos += ctxt.len();

        let next_token = logits_processor.sample(&logits)?;
        token_generated += 1;
        tokens.push(next_token);

        if Some(next_token) == eos_token_id {
            break;
        }
        if let Some(t) = tokenizer.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }
    }

    // Print the remaining decoded tokens
    if let Some(rest) = tokenizer.decode_rest().map_err(E::msg)? {
        print!("{rest}");
    }

    // Calculate and print the generation statistics
    let dt = start_gen.elapsed();
    println!(
        "\n\n{} tokens generated ({} token/s)\n",
        token_generated,
        (token_generated - 1) as f64 / dt.as_secs_f64(),
    );
    Ok(())
}
