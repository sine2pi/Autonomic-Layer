Dynamic Auto Adjustments in Transformers

The AutonomicLayer class is kind of like a central hub for dynamic auto adjustments of hyperparameters in my transformer model, featuring an in-progress meta-learner. Currently, it adjusts the base frequency (rope) and the window size of the hybrid attention block based on training loss.

Note: At the moment, the adjustment mechanism is quite aggressive (set to an extreme level).

Key Features
Dynamic Feedback: Automatically adjusts the window size of the hybrid attention block and the base frequency of the given's rotary embeddings based on the model's performance.

Hybrid Attention Block: Uses multi-head attention blocks for both small and large scales. The infrastructure for more complex adjustments is in place, but further experimentation is needed.

Work in Progress: Some of the blocks are not yet connected in this version of the model. However, the dynamic feedback mechanism is fully functional.

Hugging Face Integration: Includes a full script with Hugging Face trainer and dataset integration for ease of use. This was added for compatibility purposes, though the model performs better with a PyTorch loop.

something like this..
  
    
    import torch
    import torch.nn as nn
  
    class AutonomicLayer(nn.Module):
        def __init__(self, n_state, n_head, initial_params, max_rel_dist=1, checkpointing=False, alpha=0.001, beta=0.9):
            super(AutonomicLayer, self).__init__()
            self.params = initial_params
            self.best_loss = float('inf')
            self.base = 10000
            self.window_size = 40
            self.adjust_counter = 0
            self.factor = 1.005
            self.alpha = alpha
            self.beta = beta
            self.running_loss = None
  
      def update_base(self, new_base):
          self.base = new_base
          self.encoder.combined_rotary.update_base(self.base)
          self.decoder.combined_rotary.update_base(self.base)
  
          for name, module in self.encoder.named_modules():
              if isinstance(module, (MultiHeadAttention, CombinedRotaryEmbedding, AudioEncoder)):
                  module.update_base(self.base)
  
          for name, module in self.decoder.named_modules():
              if isinstance(module, (MultiHeadAttention, CombinedRotaryEmbedding, TextDecoder)):
                  module.update_base(self.base)
  
      def adjust_base(self, loss, factor=1.005):
          self.adjust_counter += 1  # Increment counter
          
          # Update running_loss using EWMA
          if self.running_loss is None:
              self.running_loss = loss
          else:
              self.running_loss = self.beta * self.running_loss + (1 - self.beta) * loss
  
          loss_change = loss - self.running_loss
          threshold = 0.01  # Threshold can be tuned based on your needs
  
          if loss_change < -threshold:
              new_base = self.base * factor
          elif loss_change > threshold:
              new_base = self.base / factor
          else:
              new_base = self.base
  
          self.update_base(new_base=new_base)
          self.best_loss = loss
  
          if self.adjust_counter % 200 == 0:
              print(f"Iteration {self.adjust_counter}: Adjusted base: {new_base}, Window size: {self.window_size}, Loss: {loss}")
          
          return new_base
  
      def update_window(self, new_window):
          self.window_size = new_window
  
          for name, module in self.encoder.named_modules():
              if isinstance(module, HybridAttention):
                  module.update_window(self.window_size)
  
          for name, module in self.decoder.named_modules():
              if isinstance(module, HybridAttention):
                  module.update_window(self.window_size)
  
      def adjust_window(self, loss, factor=1.005):
          self.adjust_counter += 1 
  
          if self.running_loss is None:
              self.running_loss = loss
          else:
              self.running_loss = self.beta * self.running_loss + (1 - self.beta) * loss
  
          loss_change = loss - self.running_loss
          threshold = 0.01  # Threshold can be tuned based on your needs
  
          if loss_change < -threshold:
              new_window = self.window_size * factor
          elif loss_change > threshold:
              new_window = self.window_size / factor
          else:
              new_window = self.window_size
  
          self.update_window(new_window=new_window)
          self.best_loss = loss
          
          if self.adjust_counter % 200 == 0:
              print(f"Iteration {self.adjust_counter}: Adjusted window: {new_window}, Base: {self.base}, Loss: {loss}")
          
          return new_window
  
      def forward(self, x):
          # Perform some model operations
          output = self.main_model(x)

          loss = self.calculate_loss(output)
          
          new_base = self.adjust_base(loss.item())
          new_window_size = self.adjust_window(loss.item())
          
          return output
     
    class YourModel(nn.Module):
        def __init__(self, config, autonomic_layer):
            super(YourModel, self).__init__()
            self.config = config
            self.autonomic_layer = autonomic_layer
            self.encoder = Encoder(config)
            self.decoder = Decoder(config)
  
      def forward(self, input_features, labels=None, dec_input_ids=None):
          if labels is not None:
              if dec_input_ids is None:
                  dec_input_ids = self.shift_tokens_right(
                      input_ids=labels, pad_token_id=self.config.pad_token_id, decoder_start_token_id=self.config.decoder_start_token_id
                  )
  
          encoded_features = self.encoder(input_features).to(device)
          logits = self.decoder(dec_input_ids, encoded_features)
  
          loss = None
          if labels is not None:
              loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100) 
              labels = labels.to(logits.device).long()
              loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
  
              self.autonomic_layer.adjust_base(loss.item())
              self.autonomic_layer.adjust_window(loss.item())
  
              self.autonomic_layer.update_base(self.autonomic_layer.base)
              self.autonomic_layer.update_window(self.autonomic_layer.window_size)

        return {
            "loss": loss,
            "logits": logits,
        }

"""
