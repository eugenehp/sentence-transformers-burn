use sentence_transformers::model::{
  bert_embeddings::BertEmbeddingsInferenceBatch,
  bert_model::BertModel,
};
use burn::{backend::{wgpu::WgpuDevice, Wgpu}, tensor::Tensor};
use sentence_transformers::bert_loader::{load_model_from_safetensors, load_config_from_json};
use std::env;

const BATCH_SIZE: u64 = 64;

type Backend = Wgpu;

fn main() {
  // let device = TchDevice::Mps;
  let device: WgpuDevice = WgpuDevice::BestAvailable;

  let args: Vec<String> = env::args().collect();
  let model_path = args.get(1).expect("Expected model directory as first argument");

  let config = load_config_from_json(&format!("{}/bert_config.json", model_path));
  // let model: BertModel<_> = load_model_from_safetensors::<TchBackend<f32>>(&format!("{}/bert_model.safetensors", model_path), &device, config);
  let model: BertModel<_> = load_model_from_safetensors::<Backend>(&format!("{}/bert_model.safetensors", model_path), &device, config);

  loop {
    let start = std::time::Instant::now();

    let batch = BertEmbeddingsInferenceBatch {
      tokens: Tensor::zeros(vec![BATCH_SIZE, 256], &device),
      mask_attn: Some(Tensor::ones(vec![BATCH_SIZE, 256], &device))
    };

    model.forward(batch, &device);
    
    println!("{:?}", start.elapsed());
  }
}