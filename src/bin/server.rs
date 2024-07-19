use burn::backend::wgpu::WgpuDevice;
use burn::tensor::{Tensor, Int, Data, Shape};
use sentence_transformers::model::{
  bert_embeddings::BertEmbeddingsInferenceBatch,
  bert_model::BertModel,
};
use sentence_transformers::bert_loader::{load_model_from_safetensors, load_config_from_json};
use warp::Filter;
use serde::Deserialize;
use std::sync::Arc;
use serde::Serialize;
use std::env;

use burn::backend::Wgpu;

// Type alias for the backend to use.
type Backend = Wgpu;
// type Device = WgpuDevice;

#[derive(Deserialize)]
struct EmbedRequest {
  input_ids: Vec<Vec<i32>>,
  attention_mask: Vec<Vec<i32>>,
}

#[derive(Serialize)]
struct EmbedResponse {
    embedding: Vec<Vec<Vec<f32>>>,
}

fn convert_to_3d_vec(data: &Data<f32, 3>) -> Vec<Vec<Vec<f32>>> {
  let x_dim = data.shape.dims[0];
  let y_dim = data.shape.dims[1];
  let z_dim = data.shape.dims[2];

  let mut value_iter = data.value.iter();

  (0..x_dim).map(|_| {
      (0..y_dim).map(|_| {
          (0..z_dim).map(|_| {
              *value_iter.next().expect("Unexpected tensor data size")
          }).collect()
      }).collect()
  }).collect()
}

async fn embed_handler(
  model: Arc<BertModel<Backend>,
  device: WgpuDevice,
  body: EmbedRequest
) -> Result<impl warp::Reply, warp::Rejection> {
  let batch_size = body.input_ids.len();
  let seq_length = body.input_ids.get(0).map_or(0, |v| v.len());
  let shape = Shape::new([batch_size, seq_length]);

  let input_ids = body.input_ids.into_iter().flatten().collect::<Vec<i32>>();
  let attn_mask = body.attention_mask.into_iter().flatten().collect::<Vec<i32>>();

  let input_ids_data = Data::new(input_ids, shape.clone());
  let input_ids_tensor = Tensor::<Backend, 2, Int>::from_ints(input_ids_data, &device);
  
  let attention_mask_data = Data::new(attn_mask, shape.clone());
  let attention_mask_tensor = Tensor::<Backend, 2, Int>::from_ints(attention_mask_data, &device).float();

  let input = BertEmbeddingsInferenceBatch {
    tokens: input_ids_tensor,
    mask_attn: Some(attention_mask_tensor),
  };

  let output = model.forward(input, device);
  let output_data = output.to_data();
  let embedding: Vec<Vec<Vec<f32>>> = convert_to_3d_vec(&output_data);

  let response = EmbedResponse { embedding };

  Ok(warp::reply::json(&response))
}

#[tokio::main]
async fn main() {
  // let device = TchDevice::Mps;
  let device = Default::default();
  let args: Vec<String> = env::args().collect();
  let model_path = args.get(1).expect("Expected model directory as first argument");

  let config = load_config_from_json(&format!("{}/bert_config.json", model_path));
  let model: Arc<BertModel<_>> = Arc::new(load_model_from_safetensors::<Backend>(&format!("{}/bert_model.safetensors", model_path), &device, config));

  let with_model_device = warp::any().map(move || (model.clone(), device.clone()));

  let embed = warp::post()
    .and(warp::path("embed"))
    .and(with_model_device)
    .and(warp::body::json())
    .and_then(|(model, device), body| embed_handler(model, device, body)); // Pass the Arc-wrapped model, device, and body to the handler

  println!("Listening on port 3030");
  warp::serve(embed).run(([127, 0, 0, 1], 3030)).await;
}