use burn::{
  module::{Module, Param, ParamId, ConstantRecord},
  nn::LayerNormRecord,
  nn::{LinearRecord, EmbeddingRecord},
  tensor::{backend::Backend, Tensor, Data, Shape},
};
use crate::model::{
  bert_encoder::{BertEncoderRecord, BertEncoderLayerRecord, BertOutputRecord, BertIntermediateRecord, BertAttentionRecord, BertSelfAttentionRecord, BertSelfOutputRecord}, 
  bert_embeddings::BertEmbeddingsRecord,
  bert_model::{BertModelConfig, BertModelRecord, BertModel}
};
use std::io::Read;
use std::fs::File;
use std::collections::HashMap;
use npy::NpyData;
use candle_core::{safetensors, Device, Tensor as CandleTensor};
use serde_json::Value;

fn load_npy_scalar<B: Backend>(filename: &str) -> Vec<f32> {
    // Open the file in read-only mode.
    let mut buf = vec![];

    std::fs::File::open(filename).unwrap()
        .read_to_end(&mut buf).unwrap();

    let data: NpyData<f32> = NpyData::from_bytes(&buf).unwrap();
    let data_vec: Vec<f32> = data.to_vec();
    data_vec
}

fn load_1d_tensor<B: Backend>(filename: &str, device: &B::Device) -> Param<Tensor<B, 1>> {
  // Open the file in read-only mode.
  let mut buf = vec![];

  std::fs::File::open(filename).unwrap()
      .read_to_end(&mut buf).unwrap();

  let data: NpyData<f32> = NpyData::from_bytes(&buf).unwrap();
  let data_vec: Vec<f32> = data.to_vec();

  let tensor_length = data_vec[0];
  let shape = Shape::new([tensor_length as usize]);
  let data = Data::new(data_vec[1..].to_vec(), shape);

  // Convert the loaded data into an actual 1D Tensor using from_floats
  let tensor = Tensor::<B, 1>::from_floats(data, device);

  let param = Param::initialized(ParamId::new(), tensor);
  param
}

fn load_2d_tensor<B: Backend>(filename: &str, device: &B::Device) -> Param<Tensor<B, 2>> {
  // Open the file in read-only mode.
  let mut buf = vec![];

  std::fs::File::open(filename).unwrap()
      .read_to_end(&mut buf).unwrap();

  let data: NpyData<f32> = NpyData::from_bytes(&buf).unwrap();
  let data_vec: Vec<f32> = data.to_vec();

  // Extract tensor shape from the beginning of the loaded array.
  // Here we're assuming that we've saved the tensor's shape as 2 f32 values at the start.
  let tensor_rows: usize = data_vec[0] as usize;
  let tensor_cols: usize = data_vec[1] as usize;

  // Extract tensor values
  let tensor_values_flat: Vec<f32> = data_vec[2..].to_vec(); 

  // Convert the reshaped data into an actual 2D Tensor using from_floats
  let shape = Shape::new([tensor_rows, tensor_cols]);
  let data = Data::new(tensor_values_flat, shape);
  let tensor = Tensor::<B, 2>::from_floats(data, device);

  let param = Param::initialized(ParamId::new(), tensor);
  param
}

fn load_1d_tensor_from_candle<B: Backend>(tensor: &CandleTensor, device: &B::Device) -> Tensor<B, 1> {
  let dims = tensor.dims();
  let data = tensor.to_vec1::<f32>().unwrap();
  let array: [usize; 1] = dims.try_into().expect("Unexpected size");
  let data = Data::new(data, Shape::new(array));
  let weight = Tensor::<B, 1>::from_floats(data, device);
  weight
}

fn load_2d_tensor_from_candle<B: Backend>(tensor: &CandleTensor, device: &B::Device) -> Tensor<B, 2> {
  let dims = tensor.dims();
  let data = tensor.to_vec2::<f32>().unwrap().into_iter().flatten().collect::<Vec<f32>>();
  let array: [usize; 2] = dims.try_into().expect("Unexpected size");
  let data = Data::new(data, Shape::new(array));
  let weight = Tensor::<B, 2>::from_floats(data, device);
  weight
} 

fn load_layer_norm<B: Backend>(dir: &str, device: &B::Device) -> LayerNormRecord<B> {
  let layer_norm_record = LayerNormRecord {
    beta: load_1d_tensor::<B>(&format!("{}LayerNorm_bias.npy", dir), device),
    gamma: load_1d_tensor::<B>(&format!("{}LayerNorm_weight.npy", dir), device),
    epsilon: ConstantRecord::new()
  };
  layer_norm_record
}

fn load_layer_norm_safetensor<B: Backend>(bias: &CandleTensor, weight: &CandleTensor, device: &B::Device) -> LayerNormRecord<B> {
  let beta = load_1d_tensor_from_candle::<B>(bias, device);
  let gamma = load_1d_tensor_from_candle::<B>(weight, device);

  let layer_norm_record = LayerNormRecord {
    beta: Param::initialized(ParamId::new(), beta),
    gamma: Param::initialized(ParamId::new(), gamma),
    epsilon: ConstantRecord::new()
  };
  layer_norm_record
}

fn load_linear<B: Backend>(dir: &str, device: &B::Device) -> LinearRecord<B> {
  let linear_record = LinearRecord {
    weight: load_2d_tensor::<B>(&format!("{}weight.npy", dir), device),
    bias: Some(load_1d_tensor::<B>(&format!("{}bias.npy", dir), device)),
  };
  linear_record
}

fn load_linear_safetensor<B: Backend>(bias: &CandleTensor, weight: &CandleTensor, device: &B::Device) -> LinearRecord<B> {
  let bias = load_1d_tensor_from_candle::<B>(bias, device);
  let weight = load_2d_tensor_from_candle::<B>(weight, device);

  let weight = weight.transpose();

  let linear_record = LinearRecord {
    weight: Param::initialized(ParamId::new(), weight),
    bias: Some(Param::initialized(ParamId::new(), bias)),
  };
  linear_record
}

fn load_output_layer<B: Backend>(layer_dir: &str, device: &B::Device) -> BertOutputRecord<B> {
  let output_record = BertOutputRecord {
    dense: load_linear::<B>(&format!("{}dense/", layer_dir), device),
    layer_norm: load_layer_norm::<B>(&format!("{}", layer_dir), device),
    dropout: ConstantRecord::new()
  };
  output_record
}

fn load_output_layer_safetensor<B: Backend>(dense_weight: &CandleTensor, dense_bias: &CandleTensor, layer_norm_bias: &CandleTensor, layer_norm_weight: &CandleTensor, device: &B::Device) -> BertOutputRecord<B> {
  let dense = load_linear_safetensor::<B>(dense_bias, dense_weight, device);
  let layer_norm = load_layer_norm_safetensor::<B>(layer_norm_bias, layer_norm_weight, device);

  let output_record = BertOutputRecord {
    dense,
    layer_norm,
    dropout: ConstantRecord::new()
  };
  output_record
}

fn load_intermediate_layer<B: Backend>(layer_dir: &str, device: &B::Device) -> BertIntermediateRecord<B> {
  let intermediate_record = BertIntermediateRecord {
    dense: load_linear::<B>(&format!("{}dense/", layer_dir), device),
    intermediate_act: ConstantRecord::new(),
  };
  intermediate_record
}

fn load_intermediate_layer_safetensor<B: Backend>(dense_weight: &CandleTensor, dense_bias: &CandleTensor, device: &B::Device) -> BertIntermediateRecord<B> {
  let dense = load_linear_safetensor::<B>(dense_bias, dense_weight, device);

  let intermediate_record = BertIntermediateRecord {
    dense,
    intermediate_act: ConstantRecord::new(),
  };
  
  intermediate_record
}

fn load_self_attention_layer<B: Backend>(layer_dir: &str, device: &B::Device) -> BertSelfAttentionRecord<B> {
  let attention_record = BertSelfAttentionRecord {
    query: load_linear::<B>(&format!("{}query/", layer_dir), device),
    key: load_linear::<B>(&format!("{}key/", layer_dir), device),
    value: load_linear::<B>(&format!("{}value/", layer_dir), device),
    dropout: ConstantRecord::new(),
    num_attention_heads: ConstantRecord::new(),
    attention_head_size: ConstantRecord::new(),
    all_head_size: ConstantRecord::new(),
  };
  attention_record
}

fn load_self_attention_layer_safetensor<B: Backend>(query_weight: &CandleTensor, query_bias: &CandleTensor, key_weight: &CandleTensor, key_bias: &CandleTensor, value_weight: &CandleTensor, value_bias: &CandleTensor, device: &B::Device) -> BertSelfAttentionRecord<B> {
  let query = load_linear_safetensor::<B>(query_bias, query_weight, device);
  let key = load_linear_safetensor::<B>(key_bias, key_weight, device);
  let value = load_linear_safetensor::<B>(value_bias, value_weight, device);

  let attention_record = BertSelfAttentionRecord {
    query,
    key,
    value,
    dropout: ConstantRecord::new(),
    num_attention_heads: ConstantRecord::new(),
    attention_head_size: ConstantRecord::new(),
    all_head_size: ConstantRecord::new(),
  };
  attention_record
}

fn load_self_output_layer<B: Backend>(layer_dir: &str, device: &B::Device) -> BertSelfOutputRecord<B> {
  let output_record = BertSelfOutputRecord {
    dense: load_linear::<B>(&format!("{}dense/", layer_dir), device),
    layer_norm: load_layer_norm::<B>(&format!("{}", layer_dir), device),
    dropout: ConstantRecord::new()
  };
  output_record
}

fn load_self_output_layer_safetensor<B: Backend>(dense_weight: &CandleTensor, dense_bias: &CandleTensor, layer_norm_bias: &CandleTensor, layer_norm_weight: &CandleTensor, device: &B::Device) -> BertSelfOutputRecord<B> {
  let dense = load_linear_safetensor::<B>(dense_bias, dense_weight, device);
  let layer_norm = load_layer_norm_safetensor::<B>(layer_norm_bias, layer_norm_weight, device);

  let output_record = BertSelfOutputRecord {
    dense,
    layer_norm,
    dropout: ConstantRecord::new()
  };
  output_record
}

fn load_attention_layer<B: Backend>(layer_dir: &str, device: &B::Device) -> BertAttentionRecord<B> {
  let attention_record = BertAttentionRecord {
    self_attention: load_self_attention_layer::<B>(&format!("{}self/", layer_dir), device),
    self_output: load_self_output_layer::<B>(&format!("{}output/", layer_dir), device),
  };
  attention_record
}

fn load_attention_layer_safetensor<B: Backend>(attention_tensors: HashMap<String, CandleTensor>, device: &B::Device) -> BertAttentionRecord<B> {
  let self_attention = load_self_attention_layer_safetensor::<B>(&attention_tensors["attention.self.query.weight"], &attention_tensors["attention.self.query.bias"], &attention_tensors["attention.self.key.weight"], &attention_tensors["attention.self.key.bias"], &attention_tensors["attention.self.value.weight"], &attention_tensors["attention.self.value.bias"], device);
  let self_output = load_self_output_layer_safetensor::<B>(&attention_tensors["attention.output.dense.weight"], &attention_tensors["attention.output.dense.bias"], &attention_tensors["attention.output.LayerNorm.bias"], &attention_tensors["attention.output.LayerNorm.weight"], device);

  let attention_record = BertAttentionRecord {
    self_attention,
    self_output,
  };
  attention_record
}

fn load_encoder<B: Backend>(encoder_dir: &str, device: &B::Device) -> BertEncoderRecord<B> {
  // Load n_layer
  let n_layer = load_npy_scalar::<B>(&format!("{}n_layer.npy", encoder_dir));
  let num_layers = n_layer[1] as usize;

  // Load layers
  let mut layers: Vec<BertEncoderLayerRecord<B>> = Vec::new();
  
  for i in 0..num_layers {
    let layer_dir = format!("{}layer{}/", encoder_dir, i);
    let attention_layer = load_attention_layer::<B>(format!("{}attention/", layer_dir).as_str(), device);
    let intermediate_layer = load_intermediate_layer::<B>(format!("{}intermediate/", layer_dir).as_str(), device);
    let output_layer = load_output_layer::<B>(format!("{}output/", layer_dir).as_str(), device);

    let layer_record = BertEncoderLayerRecord {
      attention: attention_layer,
      intermediate: intermediate_layer,
      output: output_layer,
    };

    layers.push(layer_record);
  }

  let encoder_record = BertEncoderRecord {
    layers,
  };

  encoder_record
}

fn load_encoder_from_safetensors<B: Backend>(encoder_tensors: HashMap<String, CandleTensor>, device: &B::Device) -> BertEncoderRecord<B> {
  // Each layer in encoder_tensors has a key like encoder.layer.0, encoder.layer.1, etc.
  // We need to extract the layers in order by iterating over the tensors and extracting the layer number
  let mut layers: HashMap<usize, HashMap<String, CandleTensor>> = HashMap::new();

  for (key, value) in encoder_tensors.iter() {
    let layer_number = key.split(".").collect::<Vec<&str>>()[2].parse::<usize>().unwrap();
    if !layers.contains_key(&layer_number) {
      layers.insert(layer_number, HashMap::new());
    }
    layers.get_mut(&layer_number).unwrap().insert(key.to_string(), value.clone());
  }

  // Sort layers.iter() by key
  let mut layers = layers.into_iter().collect::<Vec<(usize, HashMap<String, CandleTensor>)>>();
  layers.sort_by(|a, b| a.0.cmp(&b.0));

  // Now, we can iterate over the layers and load each layer
  let mut bert_encoder_layers: Vec<BertEncoderLayerRecord<B>> = Vec::new();
  for (key, value) in layers.iter() {
    let layer_key = format!("encoder.layer.{}", key.to_string());
    let attention_tensors = value.clone();
    // Remove the layer number from the key
    let attention_tensors = attention_tensors.iter().map(|(k, v)| (k.replace(&format!("{}.", layer_key), ""), v.clone())).collect::<HashMap<String, CandleTensor>>();
    let attention_layer = load_attention_layer_safetensor::<B>(attention_tensors.clone(), device);
    let intermediate_layer = load_intermediate_layer_safetensor::<B>(&value[&format!("{}.intermediate.dense.weight", layer_key)], &value[&format!("{}.intermediate.dense.bias", layer_key)], device);
    let output_layer = load_output_layer_safetensor::<B>(&value[&format!("{}.output.dense.weight", layer_key)], &value[&format!("{}.output.dense.bias", layer_key)], &value[&format!("{}.output.LayerNorm.bias", layer_key)], &value[&format!("{}.output.LayerNorm.weight", layer_key)], device);

    let layer_record = BertEncoderLayerRecord {
      attention: attention_layer,
      intermediate: intermediate_layer,
      output: output_layer,
    };

    bert_encoder_layers.push(layer_record);
  }

  let encoder_record = BertEncoderRecord {
    layers: bert_encoder_layers,
  };
  encoder_record
}

fn load_embedding<B: Backend>(embedding_dir: &str, device: &B::Device) -> EmbeddingRecord<B> {
  let embedding = EmbeddingRecord {
    weight: load_2d_tensor::<B>(&format!("{}weight.npy", embedding_dir), device),
  };

  embedding
}

fn load_embedding_safetensor<B: Backend>(weight: &CandleTensor, device: &B::Device) -> EmbeddingRecord<B> {
  let weight = load_2d_tensor_from_candle(weight, device);

  let embedding = EmbeddingRecord {
    weight: Param::initialized(ParamId::new(), weight)
  };

  embedding
}

fn load_embeddings<B: Backend>(embeddings_dir: &str, device: &B::Device) -> BertEmbeddingsRecord<B> {
  let word_embeddings = load_embedding::<B>(&format!("{}word_embeddings/", embeddings_dir), device);
  let position_embeddings = load_embedding::<B>(&format!("{}position_embeddings/", embeddings_dir), device);
  let token_type_embeddings = load_embedding::<B>(&format!("{}token_type_embeddings/", embeddings_dir), device);
  let layer_norm = load_layer_norm::<B>(&format!("{}", embeddings_dir), device);
  let dropout = ConstantRecord::new();

  let embeddings_record = BertEmbeddingsRecord {
    word_embeddings,
    position_embeddings,
    token_type_embeddings,
    layer_norm,
    dropout,
    max_position_embeddings: ConstantRecord::new(),
  };

  embeddings_record
}

fn load_embeddings_from_safetensors<B: Backend>(embedding_tensors: HashMap<String, CandleTensor>, device: &B::Device) -> BertEmbeddingsRecord<B> {
  let word_embeddings = load_embedding_safetensor(&embedding_tensors["embeddings.word_embeddings.weight"], device);
  let position_embeddings = load_embedding_safetensor(&embedding_tensors["embeddings.position_embeddings.weight"], device);
  let token_type_embeddings = load_embedding_safetensor(&embedding_tensors["embeddings.token_type_embeddings.weight"], device);
  let layer_norm = load_layer_norm_safetensor::<B>(&embedding_tensors["embeddings.LayerNorm.bias"], &embedding_tensors["embeddings.LayerNorm.weight"], device);
  let dropout = ConstantRecord::new();

  let embeddings_record = BertEmbeddingsRecord {
    word_embeddings,
    position_embeddings,
    token_type_embeddings,
    layer_norm,
    dropout,
    max_position_embeddings: ConstantRecord::new(),
  };

  embeddings_record
}

pub fn load_model<B: Backend>(dir: &str, device: &B::Device, config: BertModelConfig) -> BertModel<B> {
  let encoder_record = load_encoder::<B>(&format!("{}/encoder/", dir), device);
  let embeddings_record = load_embeddings::<B>(&format!("{}/embeddings/", dir), device);

  let model_record = BertModelRecord {
      embeddings: embeddings_record,
      encoder: encoder_record,
  };

  let mut model = config.init_with::<B>(model_record, device);

  model = model.to_device(device);
  model
}

pub fn load_model_from_safetensors<B: Backend>(file_path: &str, device: &B::Device, config: BertModelConfig) -> BertModel<B> {
  let file_path = std::path::Path::new(file_path);
  let weight_result = safetensors::load::<&std::path::Path>(file_path, &Device::Cpu);

  // Match on the result of loading the weights
  let weights = match weight_result {
    Ok(weights) => weights,
    Err(e) => panic!("Error loading weights: {:?}", e),
  };

  // Weights are stored in a HashMap<String, Tensor>
  // For each layer, it will either be prefixed with "encoder.layer." or "embeddings."
  // We need to extract both.
  let mut encoder_layers: HashMap<String, CandleTensor> = HashMap::new();
  let mut embeddings_layers: HashMap<String, CandleTensor> = HashMap::new();

  for (key, value) in weights.iter() {
    if key.starts_with("encoder.layer.") {
      encoder_layers.insert(key.to_string(), value.clone());
    } else if key.starts_with("embeddings.") {
      embeddings_layers.insert(key.to_string(), value.clone());
    }
  }
  
  let embeddings_record = load_embeddings_from_safetensors::<B>(embeddings_layers, device);
  let encoder_record = load_encoder_from_safetensors::<B>(encoder_layers, device);
  let model_record = BertModelRecord {
    embeddings: embeddings_record,
    encoder: encoder_record,
  };

  let mut model = config.init_with::<B>(model_record, device);

  model = model.to_device(device);
  model
}

pub fn load_config_from_json(json_path: &str) -> BertModelConfig {
  let file = File::open(json_path).expect("Unable to open file");
  let config: HashMap<String, Value> = serde_json::from_reader(file).expect("Unable to parse JSON");

  let model_config = BertModelConfig {
    n_heads: config["num_attention_heads"].as_i64().unwrap() as usize,
    n_layers: config["num_hidden_layers"].as_i64().unwrap() as usize,
    layer_norm_eps: config["layer_norm_eps"].as_f64().unwrap() as f64,
    hidden_size: config["hidden_size"].as_i64().unwrap() as usize,
    intermediate_size: config["intermediate_size"].as_i64().unwrap() as usize,
    hidden_act: config["hidden_act"].as_str().unwrap().to_string(),
    vocab_size: config["vocab_size"].as_i64().unwrap() as usize,
    max_position_embeddings: config["max_position_embeddings"].as_i64().unwrap() as usize,
    type_vocab_size: config["type_vocab_size"].as_i64().unwrap() as usize,
    hidden_dropout_prob: config["hidden_dropout_prob"].as_f64().unwrap() as f64,
  };

  model_config
}